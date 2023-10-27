import math
import os
from collections import namedtuple
from contextlib import closing
import random

import torch
import tqdm
import html
import datetime
import csv
import safetensors.torch

import numpy as np
from PIL import Image, PngImagePlugin
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F


from modules import shared, devices, sd_hijack, sd_models, images, sd_samplers, sd_hijack_checkpoint, errors, hashes, processing
# import modules.textual_inversion.dataset
from . import dataset as DA_dataset # here

from modules.textual_inversion.learn_schedule import LearnRateScheduler

from modules.textual_inversion.image_embedding import embedding_to_b64, embedding_from_b64, insert_image_data_embed, extract_image_data_embed, caption_image_overlay
from modules.textual_inversion.logging import save_settings_to_file

from copy import deepcopy
import json
from torch.optim.lr_scheduler import LambdaLR


from .convnext_discriminator import XPDiscriminator



TextualInversionTemplate = namedtuple("TextualInversionTemplate", ["name", "path"])
textual_inversion_templates = {}


def list_textual_inversion_templates():
    textual_inversion_templates.clear()

    for root, _, fns in os.walk(shared.cmd_opts.textual_inversion_templates_dir):
        for fn in fns:
            path = os.path.join(root, fn)

            textual_inversion_templates[fn] = TextualInversionTemplate(fn, path)

    return textual_inversion_templates


class Embedding:
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.shape = None
        self.vectors = 0
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.optimizer_state_dict = None
        self.filename = None
        self.hash = None
        self.shorthash = None

    def save(self, filename):
        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
            "sd_checkpoint": self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
        }

        torch.save(embedding_data, filename)

        if shared.opts.save_optimizer_state and self.optimizer_state_dict is not None:
            optimizer_saved_dict = {
                'hash': self.checksum(),
                'optimizer_state_dict': self.optimizer_state_dict,
            }
            torch.save(optimizer_saved_dict, f"{filename}.optim")

    def checksum(self):
        if self.cached_checksum is not None:
            return self.cached_checksum

        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        self.cached_checksum = f'{const_hash(self.vec.reshape(-1) * 100) & 0xffff:04x}'
        return self.cached_checksum

    def set_hash(self, v):
        self.hash = v
        self.shorthash = self.hash[0:12]


class DirWithTextualInversionEmbeddings:
    def __init__(self, path):
        self.path = path
        self.mtime = None

    def has_changed(self):
        if not os.path.isdir(self.path):
            return False

        mt = os.path.getmtime(self.path)
        if self.mtime is None or mt > self.mtime:
            return True

    def update(self):
        if not os.path.isdir(self.path):
            return

        self.mtime = os.path.getmtime(self.path)


class EmbeddingDatabase:
    def __init__(self):
        self.ids_lookup = {}
        self.word_embeddings = {}
        self.skipped_embeddings = {}
        self.expected_shape = -1
        self.embedding_dirs = {}
        self.previously_displayed_embeddings = ()

        self.dir_mtime = None # HERE

    def add_embedding_dir(self, path):
        self.embedding_dirs[path] = DirWithTextualInversionEmbeddings(path)

    def clear_embedding_dirs(self):
        self.embedding_dirs.clear()

    def register_embedding(self, embedding, model):
        """
        HERE
        """

        self.word_embeddings[embedding.name] = embedding

        ids = model.cond_stage_model.tokenizer([embedding.name], add_special_tokens=False)['input_ids'][0]

        first_id = ids[0]
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []

        self.ids_lookup[first_id] = sorted(self.ids_lookup[first_id] + [(ids, embedding)], key=lambda x: len(x[0]), reverse=True)

        return embedding

    def register_embedding_by_name(self, embedding, model, name):
        ids = model.cond_stage_model.tokenize([name])[0]
        first_id = ids[0]
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []
        if name in self.word_embeddings:
            # remove old one from the lookup list
            lookup = [x for x in self.ids_lookup[first_id] if x[1].name!=name]
        else:
            lookup = self.ids_lookup[first_id]
        if embedding is not None:
            lookup += [(ids, embedding)]
        self.ids_lookup[first_id] = sorted(lookup, key=lambda x: len(x[0]), reverse=True)
        if embedding is None:
            # unregister embedding with specified name
            if name in self.word_embeddings:
                del self.word_embeddings[name]
            if len(self.ids_lookup[first_id])==0:
                del self.ids_lookup[first_id]
            return None
        self.word_embeddings[name] = embedding
        return embedding

    def get_expected_shape(self):
        vec = shared.sd_model.cond_stage_model.encode_embedding_init_text(",", 1)
        return vec.shape[1]

    def load_from_file(self, path, filename):
        name, ext = os.path.splitext(filename)
        ext = ext.upper()

        if ext in ['.PNG', '.WEBP', '.JXL', '.AVIF']:
            _, second_ext = os.path.splitext(name)
            if second_ext.upper() == '.PREVIEW':
                return

            embed_image = Image.open(path)
            if hasattr(embed_image, 'text') and 'sd-ti-embedding' in embed_image.text:
                data = embedding_from_b64(embed_image.text['sd-ti-embedding'])
                name = data.get('name', name)
            else:
                data = extract_image_data_embed(embed_image)
                if data:
                    name = data.get('name', name)
                else:
                    # if data is None, means this is not an embeding, just a preview image
                    return
        elif ext in ['.BIN', '.PT']:
            data = torch.load(path, map_location="cpu")
        elif ext in ['.SAFETENSORS']:
            data = safetensors.torch.load_file(path, device="cpu")
        else:
            return


        # textual inversion embeddings
        if 'string_to_param' in data:
            param_dict = data['string_to_param']
            param_dict = getattr(param_dict, '_parameters', param_dict)  # fix for torch 1.12.1 loading saved file from torch 1.11
            assert len(param_dict) == 1, 'embedding file has multiple terms in it'
            emb = next(iter(param_dict.items()))[1]
            vec = emb.detach().to(devices.device, dtype=torch.float32)
            shape = vec.shape[-1]
            vectors = vec.shape[0]
        elif type(data) == dict and 'clip_g' in data and 'clip_l' in data:  # SDXL embedding
            vec = {k: v.detach().to(devices.device, dtype=torch.float32) for k, v in data.items()}
            shape = data['clip_g'].shape[-1] + data['clip_l'].shape[-1]
            vectors = data['clip_g'].shape[0]
        elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor: # diffuser concepts
            assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

            emb = next(iter(data.values()))
            if len(emb.shape) == 1:
                emb = emb.unsqueeze(0)
            vec = emb.detach().to(devices.device, dtype=torch.float32)
            shape = vec.shape[-1]
            vectors = vec.shape[0]
        else:
            raise Exception(f"Couldn't identify {filename} as neither textual inversion embedding nor diffuser concept.")

        embedding = Embedding(vec, name)
        embedding.step = data.get('step', None)
        embedding.sd_checkpoint = data.get('sd_checkpoint', None)
        embedding.sd_checkpoint_name = data.get('sd_checkpoint_name', None)
        embedding.vectors = vectors
        embedding.shape = shape
        embedding.filename = path
        embedding.set_hash(hashes.sha256(embedding.filename, "textual_inversion/" + name) or '')

        if self.expected_shape == -1 or self.expected_shape == embedding.shape:
            self.register_embedding(embedding, shared.sd_model)
        else:
            self.skipped_embeddings[name] = embedding

    def load_from_dir(self, embdir):
        if not os.path.isdir(embdir.path):
            return

        for root, _, fns in os.walk(embdir.path, followlinks=True):
            for fn in fns:
                try:
                    fullfn = os.path.join(root, fn)

                    if os.stat(fullfn).st_size == 0:
                        continue

                    self.load_from_file(fullfn, fn)
                except Exception:
                    errors.report(f"Error loading embedding {fn}", exc_info=True)
                    continue

    def load_textual_inversion_embeddings(self, force_reload=False):
        if not force_reload:
            need_reload = False
            for embdir in self.embedding_dirs.values():
                if embdir.has_changed():
                    need_reload = True
                    break

            if not need_reload:
                return

        self.ids_lookup.clear()
        self.word_embeddings.clear()
        self.skipped_embeddings.clear()
        self.expected_shape = self.get_expected_shape()

        for embdir in self.embedding_dirs.values():
            self.load_from_dir(embdir)
            embdir.update()

        # re-sort word_embeddings because load_from_dir may not load in alphabetic order.
        # using a temporary copy so we don't reinitialize self.word_embeddings in case other objects have a reference to it.
        sorted_word_embeddings = {e.name: e for e in sorted(self.word_embeddings.values(), key=lambda e: e.name.lower())}
        self.word_embeddings.clear()
        self.word_embeddings.update(sorted_word_embeddings)

        displayed_embeddings = (tuple(self.word_embeddings.keys()), tuple(self.skipped_embeddings.keys()))
        if shared.opts.textual_inversion_print_at_load and self.previously_displayed_embeddings != displayed_embeddings:
            self.previously_displayed_embeddings = displayed_embeddings
            print(f"Textual inversion embeddings loaded({len(self.word_embeddings)}): {', '.join(self.word_embeddings.keys())}")
            if self.skipped_embeddings:
                print(f"Textual inversion embeddings skipped({len(self.skipped_embeddings)}): {', '.join(self.skipped_embeddings.keys())}")

    def find_embedding_at_position(self, tokens, offset):
        """
        HERE
        """
        token = tokens[offset]
        possible_matches = self.ids_lookup.get(token, None)

        if possible_matches is None:
            return None, None

        for ids, embedding in possible_matches:
            if tokens[offset:offset + len(ids)] == ids:
                return embedding, len(ids)

        return None, None

    def load_words_embeddings(self):
        """
        HERE
        """
        mt = os.path.getmtime(self.embeddings_dir)
        if self.dir_mtime is not None and mt <= self.dir_mtime:
            return

        self.dir_mtime = mt
        self.ids_lookup.clear()
        self.word_embeddings.clear()

        def process_file(path, filename):
            name = os.path.splitext(filename)[0]

            data = []

            if os.path.splitext(filename.upper())[-1] in ['.PNG', '.WEBP', '.JXL', '.AVIF']:
                embed_image = Image.open(path)
                if hasattr(embed_image, 'text') and 'sd-ti-embedding' in embed_image.text:
                    data = embedding_from_b64(embed_image.text['sd-ti-embedding'])
                    name = data.get('name', name)
                else:
                    data = extract_image_data_embed(embed_image)
                    name = data.get('name', name)
            else:
                data = torch.load(path, map_location="cpu")

            # pseudo-words embeddings
            if 'string_to_param' in data:
                param_dict = data['string_to_param']
                if hasattr(param_dict, '_parameters'):
                    param_dict = getattr(param_dict, '_parameters')  # fix for torch 1.12.1 loading saved file from torch 1.11
                assert len(param_dict) == 1, 'embedding file has multiple terms in it'
                emb = next(iter(param_dict.items()))[1]
            # diffuser concepts
            elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
                assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

                emb = next(iter(data.values()))
                if len(emb.shape) == 1:
                    emb = emb.unsqueeze(0)
            else:
                raise Exception(f"Couldn't identify {filename} as neither words embedding nor diffuser concept.")

            vec = emb.detach().to(devices.device, dtype=torch.float32)
            embedding = Embedding(vec, name)
            embedding.step = data.get('step', None)
            embedding.sd_checkpoint = data.get('sd_checkpoint', None)
            embedding.sd_checkpoint_name = data.get('sd_checkpoint_name', None)
            self.register_embedding(embedding, shared.sd_model)

        for fn in os.listdir(self.embeddings_dir):
            try:
                fullfn = os.path.join(self.embeddings_dir, fn)

                if os.stat(fullfn).st_size == 0:
                    continue

                process_file(fullfn, fn)
            except Exception:
                print(f"Error loading emedding {fn}:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                continue

        print(f"Loaded a total of {len(self.word_embeddings)} words embeddings.")
        print("Embeddings:", ', '.join(self.word_embeddings.keys()))



def create_embedding(name, num_vectors_per_token, overwrite_old, init_text='*'):
    """
    HERE
    """
    cond_model = shared.sd_model.cond_stage_model
    embedding_layer = cond_model.wrapped.transformer.text_model.embeddings

    with devices.autocast():
        cond_model([""])  # will send cond model to GPU if lowvram/medvram is active

    ids = cond_model.tokenizer(init_text, max_length=num_vectors_per_token, return_tensors="pt", add_special_tokens=False)["input_ids"]
    embedded = embedding_layer.token_embedding.wrapped(ids.to(devices.device)).squeeze(0)
    vec = torch.zeros((num_vectors_per_token, embedded.shape[1]), device=devices.device)

    for i in range(num_vectors_per_token):
        vec[i] = embedded[i * int(embedded.shape[0]) // num_vectors_per_token]
        if '-neg' in name:
            vec[i]+=torch.randn_like(vec[i])*1e-3

    # Remove illegal characters from name.
    name = "".join( x for x in name if (x.isalnum() or x in "._- "))
    fn = os.path.join(shared.cmd_opts.embeddings_dir, f"{name}.pt")
    if not overwrite_old:
        assert not os.path.exists(fn), f"file {fn} already exists"

    embedding = Embedding(vec, name)
    embedding.step = 0
    embedding.save(fn)

    return fn


def write_loss(log_directory, filename, step, epoch_len, values):
    if shared.opts.training_write_csv_every == 0:
        return

    if step % shared.opts.training_write_csv_every != 0:
        return
    write_csv_header = False if os.path.exists(os.path.join(log_directory, filename)) else True

    with open(os.path.join(log_directory, filename), "a+", newline='') as fout:
        csv_writer = csv.DictWriter(fout, fieldnames=["step", "epoch", "epoch_step", *(values.keys())])

        if write_csv_header:
            csv_writer.writeheader()

        epoch = (step - 1) // epoch_len
        epoch_step = (step - 1) % epoch_len

        csv_writer.writerow({
            "step": step,
            "epoch": epoch,
            "epoch_step": epoch_step,
            **values,
        })

def tensorboard_setup(log_directory):
    os.makedirs(os.path.join(log_directory, "tensorboard"), exist_ok=True)
    return SummaryWriter(
            log_dir=os.path.join(log_directory, "tensorboard"),
            flush_secs=shared.opts.training_tensorboard_flush_every)

def tensorboard_add(tensorboard_writer, loss, global_step, step, learn_rate, epoch_num):
    tensorboard_add_scaler(tensorboard_writer, "Loss/train", loss, global_step)
    tensorboard_add_scaler(tensorboard_writer, f"Loss/train/epoch-{epoch_num}", loss, step)
    tensorboard_add_scaler(tensorboard_writer, "Learn rate/train", learn_rate, global_step)
    tensorboard_add_scaler(tensorboard_writer, f"Learn rate/train/epoch-{epoch_num}", learn_rate, step)

def tensorboard_add_scaler(tensorboard_writer, tag, value, step):
    tensorboard_writer.add_scalar(tag=tag,
        scalar_value=value, global_step=step)

def tensorboard_add_image(tensorboard_writer, tag, pil_image, step):
    # Convert a pil image to a torch tensor
    img_tensor = torch.as_tensor(np.array(pil_image, copy=True))
    img_tensor = img_tensor.view(pil_image.size[1], pil_image.size[0],
        len(pil_image.getbands()))
    img_tensor = img_tensor.permute((2, 0, 1))

    tensorboard_writer.add_image(tag, img_tensor, global_step=step)

# def validate_train_inputs(model_name, learn_rate, batch_size, gradient_step, data_root, template_file, template_filename, steps, save_model_every, create_image_every, log_directory, name="embedding"):
#     assert model_name, f"{name} not selected"
#     assert learn_rate, "Learning rate is empty or 0"
#     assert isinstance(batch_size, int), "Batch size must be integer"
#     assert batch_size > 0, "Batch size must be positive"
#     assert isinstance(gradient_step, int), "Gradient accumulation step must be integer"
#     assert gradient_step > 0, "Gradient accumulation step must be positive"
#     assert data_root, "Dataset directory is empty"
#     assert os.path.isdir(data_root), "Dataset directory doesn't exist"
#     assert os.listdir(data_root), "Dataset directory is empty"
#     assert template_filename, "Prompt template file not selected"
#     assert template_file, f"Prompt template file {template_filename} not found"
#     assert os.path.isfile(template_file.path), f"Prompt template file {template_filename} doesn't exist"
#     assert steps, "Max steps is empty or 0"
#     assert isinstance(steps, int), "Max steps must be integer"
#     assert steps > 0, "Max steps must be positive"
#     assert isinstance(save_model_every, int), "Save {name} must be integer"
#     assert save_model_every >= 0, "Save {name} must be positive or 0"
#     assert isinstance(create_image_every, int), "Create image must be integer"
#     assert create_image_every >= 0, "Create image must be positive or 0"
#     if save_model_every or create_image_every:
#         assert log_directory, "Log directory is empty"

# HERE

# def train_embedding(id_task, embedding_name, learn_rate, batch_size, gradient_step, data_root, log_directory, training_width, training_height, varsize, steps, clip_grad_mode, clip_grad_value, shuffle_tags, tag_drop_out, latent_sampling_method, use_weight, create_image_every, save_embedding_every, template_filename, save_image_with_stored_embedding, preview_from_txt2img, preview_prompt, preview_negative_prompt, preview_steps, preview_sampler_index, preview_cfg_scale, preview_seed, preview_width, preview_height):
#     from modules import processing

#     save_embedding_every = save_embedding_every or 0
#     create_image_every = create_image_every or 0
#     template_file = textual_inversion_templates.get(template_filename, None)
#     validate_train_inputs(embedding_name, learn_rate, batch_size, gradient_step, data_root, template_file, template_filename, steps, save_embedding_every, create_image_every, log_directory, name="embedding")
#     template_file = template_file.path

#     shared.state.job = "train-embedding"
#     shared.state.textinfo = "Initializing textual inversion training..."
#     shared.state.job_count = steps

#     filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')

#     log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), embedding_name)
#     unload = shared.opts.unload_models_when_training

#     if save_embedding_every > 0:
#         embedding_dir = os.path.join(log_directory, "embeddings")
#         os.makedirs(embedding_dir, exist_ok=True)
#     else:
#         embedding_dir = None

#     if create_image_every > 0:
#         images_dir = os.path.join(log_directory, "images")
#         os.makedirs(images_dir, exist_ok=True)
#     else:
#         images_dir = None

#     if create_image_every > 0 and save_image_with_stored_embedding:
#         images_embeds_dir = os.path.join(log_directory, "image_embeddings")
#         os.makedirs(images_embeds_dir, exist_ok=True)
#     else:
#         images_embeds_dir = None

#     hijack = sd_hijack.model_hijack

#     embedding = hijack.embedding_db.word_embeddings[embedding_name]
#     checkpoint = sd_models.select_checkpoint()

#     initial_step = embedding.step or 0
#     if initial_step >= steps:
#         shared.state.textinfo = "Model has already been trained beyond specified max steps"
#         return embedding, filename

#     scheduler = LearnRateScheduler(learn_rate, steps, initial_step)
#     clip_grad = torch.nn.utils.clip_grad_value_ if clip_grad_mode == "value" else \
#         torch.nn.utils.clip_grad_norm_ if clip_grad_mode == "norm" else \
#         None
#     if clip_grad:
#         clip_grad_sched = LearnRateScheduler(clip_grad_value, steps, initial_step, verbose=False)
#     # dataset loading may take a while, so input validations and early returns should be done before this
#     shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
#     old_parallel_processing_allowed = shared.parallel_processing_allowed

#     if shared.opts.training_enable_tensorboard:
#         tensorboard_writer = tensorboard_setup(log_directory)

#     pin_memory = shared.opts.pin_memory

#     ds = modules.textual_inversion.dataset.PersonalizedBase(data_root=data_root, width=training_width, height=training_height, repeats=shared.opts.training_image_repeats_per_epoch, placeholder_token=embedding_name, model=shared.sd_model, cond_model=shared.sd_model.cond_stage_model, device=devices.device, template_file=template_file, batch_size=batch_size, gradient_step=gradient_step, shuffle_tags=shuffle_tags, tag_drop_out=tag_drop_out, latent_sampling_method=latent_sampling_method, varsize=varsize, use_weight=use_weight)

#     if shared.opts.save_training_settings_to_txt:
#         save_settings_to_file(log_directory, {**dict(model_name=checkpoint.model_name, model_hash=checkpoint.shorthash, num_of_dataset_images=len(ds), num_vectors_per_token=len(embedding.vec)), **locals()})

#     latent_sampling_method = ds.latent_sampling_method

#     dl = modules.textual_inversion.dataset.PersonalizedDataLoader(ds, latent_sampling_method=latent_sampling_method, batch_size=ds.batch_size, pin_memory=pin_memory)

#     if unload:
#         shared.parallel_processing_allowed = False
#         shared.sd_model.first_stage_model.to(devices.cpu)

#     embedding.vec.requires_grad = True
#     optimizer = torch.optim.AdamW([embedding.vec], lr=scheduler.learn_rate, weight_decay=0.0)
#     if shared.opts.save_optimizer_state:
#         optimizer_state_dict = None
#         if os.path.exists(f"{filename}.optim"):
#             optimizer_saved_dict = torch.load(f"{filename}.optim", map_location='cpu')
#             if embedding.checksum() == optimizer_saved_dict.get('hash', None):
#                 optimizer_state_dict = optimizer_saved_dict.get('optimizer_state_dict', None)

#         if optimizer_state_dict is not None:
#             optimizer.load_state_dict(optimizer_state_dict)
#             print("Loaded existing optimizer from checkpoint")
#         else:
#             print("No saved optimizer exists in checkpoint")

#     scaler = torch.cuda.amp.GradScaler()

#     batch_size = ds.batch_size
#     gradient_step = ds.gradient_step
#     # n steps = batch_size * gradient_step * n image processed
#     steps_per_epoch = len(ds) // batch_size // gradient_step
#     max_steps_per_epoch = len(ds) // batch_size - (len(ds) // batch_size) % gradient_step
#     loss_step = 0
#     _loss_step = 0 #internal

#     last_saved_file = "<none>"
#     last_saved_image = "<none>"
#     forced_filename = "<none>"
#     embedding_yet_to_be_embedded = False

#     is_training_inpainting_model = shared.sd_model.model.conditioning_key in {'hybrid', 'concat'}
#     img_c = None

#     pbar = tqdm.tqdm(total=steps - initial_step)
#     try:
#         sd_hijack_checkpoint.add()

#         for _ in range((steps-initial_step) * gradient_step):
#             if scheduler.finished:
#                 break
#             if shared.state.interrupted:
#                 break
#             for j, batch in enumerate(dl):
#                 # works as a drop_last=True for gradient accumulation
#                 if j == max_steps_per_epoch:
#                     break
#                 scheduler.apply(optimizer, embedding.step)
#                 if scheduler.finished:
#                     break
#                 if shared.state.interrupted:
#                     break

#                 if clip_grad:
#                     clip_grad_sched.step(embedding.step)

#                 with devices.autocast():
#                     x = batch.latent_sample.to(devices.device, non_blocking=pin_memory)
#                     if use_weight:
#                         w = batch.weight.to(devices.device, non_blocking=pin_memory)
#                     c = shared.sd_model.cond_stage_model(batch.cond_text)

#                     if is_training_inpainting_model:
#                         if img_c is None:
#                             img_c = processing.txt2img_image_conditioning(shared.sd_model, c, training_width, training_height)

#                         cond = {"c_concat": [img_c], "c_crossattn": [c]}
#                     else:
#                         cond = c

#                     if use_weight:
#                         loss = shared.sd_model.weighted_forward(x, cond, w)[0] / gradient_step
#                         del w
#                     else:
#                         loss = shared.sd_model.forward(x, cond)[0] / gradient_step
#                     del x

#                     _loss_step += loss.item()
#                 scaler.scale(loss).backward()

#                 # go back until we reach gradient accumulation steps
#                 if (j + 1) % gradient_step != 0:
#                     continue

#                 if clip_grad:
#                     clip_grad(embedding.vec, clip_grad_sched.learn_rate)

#                 scaler.step(optimizer)
#                 scaler.update()
#                 embedding.step += 1
#                 pbar.update()
#                 optimizer.zero_grad(set_to_none=True)
#                 loss_step = _loss_step
#                 _loss_step = 0

#                 steps_done = embedding.step + 1

#                 epoch_num = embedding.step // steps_per_epoch
#                 epoch_step = embedding.step % steps_per_epoch

#                 description = f"Training textual inversion [Epoch {epoch_num}: {epoch_step+1}/{steps_per_epoch}] loss: {loss_step:.7f}"
#                 pbar.set_description(description)
#                 if embedding_dir is not None and steps_done % save_embedding_every == 0:
#                     # Before saving, change name to match current checkpoint.
#                     embedding_name_every = f'{embedding_name}-{steps_done}'
#                     last_saved_file = os.path.join(embedding_dir, f'{embedding_name_every}.pt')
#                     save_embedding(embedding, optimizer, checkpoint, embedding_name_every, last_saved_file, remove_cached_checksum=True)
#                     embedding_yet_to_be_embedded = True

#                 write_loss(log_directory, "textual_inversion_loss.csv", embedding.step, steps_per_epoch, {
#                     "loss": f"{loss_step:.7f}",
#                     "learn_rate": scheduler.learn_rate
#                 })

#                 if images_dir is not None and steps_done % create_image_every == 0:
#                     forced_filename = f'{embedding_name}-{steps_done}'
#                     last_saved_image = os.path.join(images_dir, forced_filename)

#                     shared.sd_model.first_stage_model.to(devices.device)

#                     p = processing.StableDiffusionProcessingTxt2Img(
#                         sd_model=shared.sd_model,
#                         do_not_save_grid=True,
#                         do_not_save_samples=True,
#                         do_not_reload_embeddings=True,
#                     )

#                     if preview_from_txt2img:
#                         p.prompt = preview_prompt
#                         p.negative_prompt = preview_negative_prompt
#                         p.steps = preview_steps
#                         p.sampler_name = sd_samplers.samplers[preview_sampler_index].name
#                         p.cfg_scale = preview_cfg_scale
#                         p.seed = preview_seed
#                         p.width = preview_width
#                         p.height = preview_height
#                     else:
#                         p.prompt = batch.cond_text[0]
#                         p.steps = 20
#                         p.width = training_width
#                         p.height = training_height

#                     preview_text = p.prompt

#                     with closing(p):
#                         processed = processing.process_images(p)
#                         image = processed.images[0] if len(processed.images) > 0 else None

#                     if unload:
#                         shared.sd_model.first_stage_model.to(devices.cpu)

#                     if image is not None:
#                         shared.state.assign_current_image(image)

#                         last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt, shared.opts.samples_format, processed.infotexts[0], p=p, forced_filename=forced_filename, save_to_dirs=False)
#                         last_saved_image += f", prompt: {preview_text}"

#                         if shared.opts.training_enable_tensorboard and shared.opts.training_tensorboard_save_images:
#                             tensorboard_add_image(tensorboard_writer, f"Validation at epoch {epoch_num}", image, embedding.step)

#                     if save_image_with_stored_embedding and os.path.exists(last_saved_file) and embedding_yet_to_be_embedded:

#                         last_saved_image_chunks = os.path.join(images_embeds_dir, f'{embedding_name}-{steps_done}.png')

#                         info = PngImagePlugin.PngInfo()
#                         data = torch.load(last_saved_file)
#                         info.add_text("sd-ti-embedding", embedding_to_b64(data))

#                         title = f"<{data.get('name', '???')}>"

#                         try:
#                             vectorSize = list(data['string_to_param'].values())[0].shape[0]
#                         except Exception:
#                             vectorSize = '?'

#                         checkpoint = sd_models.select_checkpoint()
#                         footer_left = checkpoint.model_name
#                         footer_mid = f'[{checkpoint.shorthash}]'
#                         footer_right = f'{vectorSize}v {steps_done}s'

#                         captioned_image = caption_image_overlay(image, title, footer_left, footer_mid, footer_right)
#                         captioned_image = insert_image_data_embed(captioned_image, data)

#                         captioned_image.save(last_saved_image_chunks, "PNG", pnginfo=info)
#                         embedding_yet_to_be_embedded = False

#                     last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt, shared.opts.samples_format, processed.infotexts[0], p=p, forced_filename=forced_filename, save_to_dirs=False)
#                     last_saved_image += f", prompt: {preview_text}"

#                 shared.state.job_no = embedding.step

#                 shared.state.textinfo = f"""
# <p>
# Loss: {loss_step:.7f}<br/>
# Step: {steps_done}<br/>
# Last prompt: {html.escape(batch.cond_text[0])}<br/>
# Last saved embedding: {html.escape(last_saved_file)}<br/>
# Last saved image: {html.escape(last_saved_image)}<br/>
# </p>
# """
#         filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')
#         save_embedding(embedding, optimizer, checkpoint, embedding_name, filename, remove_cached_checksum=True)
#     except Exception:
#         errors.report("Error training embedding", exc_info=True)
#     finally:
#         pbar.leave = False
#         pbar.close()
#         shared.sd_model.first_stage_model.to(devices.device)
#         shared.parallel_processing_allowed = old_parallel_processing_allowed
#         sd_hijack_checkpoint.remove()

#     return embedding, filename

def set_seed(seed):
    """
    HERE
    """
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms


def validate_train_inputs(model_name, learn_rate, batch_size, data_root, template_file, steps, save_model_every, create_image_every, log_directory, name="embedding"):
    assert model_name, f"{name} not selected"
    assert learn_rate, "Learning rate is empty or 0"
    assert isinstance(batch_size, int), "Batch size must be integer"
    assert batch_size > 0, "Batch size must be positive"
    assert data_root, "Dataset directory is empty"
    assert os.path.isdir(data_root), "Dataset directory doesn't exist"
    assert os.listdir(data_root), "Dataset directory is empty"
    assert template_file, "Prompt template file is empty"
    assert os.path.isfile(template_file), "Prompt template file doesn't exist" # HERE
    assert steps, "Max steps is empty or 0"
    assert isinstance(steps, int), "Max steps must be integer"
    assert steps > 0 , "Max steps must be positive"
    assert isinstance(save_model_every, int), "Save {name} must be integer"
    assert save_model_every >= 0 , "Save {name} must be positive or 0"
    assert isinstance(create_image_every, int), "Create image must be integer"
    assert create_image_every >= 0 , "Create image must be positive or 0"
    if save_model_every or create_image_every:
        assert log_directory, "Log directory is empty"


from ldm.util import default
from ldm.modules.diffusionmodules.util import extract_into_tensor

#a_t=0.005
#sqrt_one_minus_at=np.sqrt(1.-a_t)
def p_losses_hook(x_start, cond, t, noise=None, scale=(1.0,1.0), att_mask=None, dy_cfg_f='ln'):
    self=shared.sd_model
    noise = default(noise, lambda: torch.randn_like(x_start))
    x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

    # support negative prompt tuning
    t_raw = t
    x_noisy_raw = x_noisy
    if scale[1] != 1.0:
        x_noisy = torch.cat([x_noisy] * 2)
        t = torch.cat([t] * 2)

    model_output = self.apply_model(x_noisy, t, cond)

    # support negative prompt tuning
    if scale[1] != 1.0:
        e_t_uncond, e_t = model_output.chunk(2)
        if scale[0] != scale[1]:
            rate = t_raw / (self.num_timesteps - 1)
            if dy_cfg_f=='cos':
                rate = torch.cos((rate-1)*math.pi/2)
            elif dy_cfg_f=='cos2':
                rate = 1-torch.cos(rate*math.pi/2)
            elif dy_cfg_f=='ln':
                pass
            else:
                rate = eval(dy_cfg_f)
        else:
            rate = 1
        model_output = e_t_uncond + ((scale[1]-scale[0])*rate+scale[0]) * (e_t - e_t_uncond)

    loss_dict = {}
    prefix = 'train' if self.training else 'val'

    if self.parameterization == "x0":
        target = x_start
    elif self.parameterization == "eps":
        target = noise
    else:
        raise NotImplementedError()

    loss_simple = self.get_loss(model_output, target, mean=False)
    if att_mask is not None:
        loss_simple=loss_simple*att_mask
    loss_simple=loss_simple.mean([1, 2, 3])
    loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

    logvar_t = self.logvar[t_raw].to(self.device)
    loss = loss_simple / torch.exp(logvar_t) + logvar_t
    # loss = loss_simple / torch.exp(self.logvar) + self.logvar
    if self.learn_logvar:
        loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
        loss_dict.update({'logvar': self.logvar.data.mean()})

    loss = self.l_simple_weight * loss.mean()

    loss_vlb = self.get_loss(model_output, target, mean=False)
    if att_mask is not None:
        loss_vlb=loss_vlb*att_mask
    loss_vlb=loss_vlb.mean(dim=(1, 2, 3))
    loss_vlb = (self.lvlb_weights[t_raw] * loss_vlb).mean()
    loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
    loss += (self.original_elbo_weight * loss_vlb)
    loss_dict.update({f'{prefix}/loss': loss})

    img = (x_noisy_raw-extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t_raw, x_start.shape) * model_output)/extract_into_tensor(self.sqrt_alphas_cumprod, t_raw, x_start.shape)

    return loss, loss_dict, img

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        current_params, ma_params = current_model.vec, ma_model.vec
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def get_cfg_range(cfg_text:str):
    dy_cfg_f='ln'
    if cfg_text.find(':')!=-1:
        cfg_text, dy_cfg_f = cfg_text.split(':')

    if cfg_text.find('-')!=-1:
        l, h = cfg_text.split('-')
        return float(l), float(h), dy_cfg_f
    else:
        return float(cfg_text), float(cfg_text), dy_cfg_f


def train_embedding(id_task,
                    embedding_name,
                    learn_rate,
                    batch_size,
                    data_root,
                    log_directory,
                    training_width,
                    training_height,
                    steps,
                    create_image_every,
                    save_embedding_every,
                    template_file,
                    save_image_with_stored_embedding,
                    # cfg_scale,
                    preview_from_txt2img,
                    preview_prompt=None, preview_negative_prompt=None, preview_steps=None, preview_sampler_index=None, preview_cfg_scale=None, preview_seed=None, preview_width=None, preview_height = None,
                    seed = 114514,
                    classifier_path = "",
                    use_negative = True, use_att_map = True, use_rec = True,
                    # neg_train = True,
                    # att_map = True,
                    # rec_train = False,
                    rec_loss_w = 1.0,
                    neg_lr_w = 1.0,
                    ema_w = 1.0,
                    ema_rep_step = 25,
                    ema_w_neg = 1.0,
                    ema_rep_step_neg = 25,
                    adam_beta1 = 0.9,
                    adam_beta2 = 0.999,
                    fw_pos_only = False,
                    accumulation_steps = 1,
                    unet_train = False,
                    unet_lr = 0.000005,
                    cfg_scale = "3.0", # here 3.0
                    ):
    # NOTE: the first `_` argument is the TaskID, which is not used in this function
    # but is required by Automatic1111 to set the TaskID for the thread.
    """
    HERE
    """
    set_seed(seed)
    template_file = "textual_inversion_templates/" + template_file

    save_embedding_every = save_embedding_every or 0
    create_image_every = create_image_every or 0
    validate_train_inputs(embedding_name, learn_rate, batch_size, data_root, template_file, steps, save_embedding_every, create_image_every, log_directory, name="embedding")

    p_losses_backup = shared.sd_model.p_losses
    shared.sd_model.p_losses = p_losses_hook  # hook p_losses

    #maybe fix issue #1
    shared.sd_model.first_stage_model.to(devices.device)

    shared.state.textinfo = "Initializing prompt tuning..."
    shared.state.job_count = steps

    filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')

    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), embedding_name)
    unload = False #shared.opts.unload_models_when_training

    if save_embedding_every > 0:
        embedding_dir = os.path.join(log_directory, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
    else:
        embedding_dir = None

    if create_image_every > 0:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    if create_image_every > 0 and save_image_with_stored_embedding:
        images_embeds_dir = os.path.join(log_directory, "image_embeddings")
        os.makedirs(images_embeds_dir, exist_ok=True)
    else:
        images_embeds_dir = None

    cond_model = shared.sd_model.cond_stage_model

    hijack = sd_hijack.model_hijack

    embedding = hijack.embedding_db.word_embeddings[embedding_name]
    checkpoint = sd_models.select_checkpoint()

    ititial_step = embedding.step or 0
    if ititial_step >= steps:
        shared.state.textinfo = f"Model has already been trained beyond specified max steps"
        return embedding, filename

    scheduler = LearnRateScheduler(learn_rate, steps, ititial_step)

    # dataset loading may take a while, so input validations and early returns should be done before this
    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}... {embedding_name}"
    with torch.autocast("cuda"):
        ds = DA_dataset.PersonalizedBase(data_root=data_root, width=training_width, height=training_height, repeats=shared.opts.training_image_repeats_per_epoch, placeholder_token=embedding_name, model=shared.sd_model, device=devices.device, template_file=template_file, batch_size=batch_size, fw_pos_only=fw_pos_only)
    if unload:
        shared.sd_model.first_stage_model.to(devices.cpu)

    ema = EMA(ema_w)
    ema_neg = EMA(ema_w_neg)

    embedding_ema = deepcopy(embedding)

    embedding.vec.requires_grad = True
    if use_negative:
        embedding_neg = hijack.embedding_db.word_embeddings[embedding_name + '-neg']  # negative prompt embeddings
        embedding_neg_ema = deepcopy(embedding_neg)

        embedding_neg.vec.requires_grad = True

    hyper_param = {
        'lr': learn_rate,
        'bs': batch_size,
        'cfg': cfg_scale,
        'size': [training_width, training_height],
        'neg': use_negative,
        'rec': use_rec,
        'seed': seed,
        'prompt_len': embedding.vec.shape,
    }
    if use_negative:
        hyper_param['prompt_len_neg'] = embedding_neg.vec.shape
        hyper_param['neg_lr_w'] = neg_lr_w
    if use_rec:
        hyper_param['rec_loss_w'] = rec_loss_w

    hyper_param = json.dumps(hyper_param, sort_keys=True, indent=4)
    with open(os.path.join(log_directory, 'hyper_param.json'), 'w') as f:
        f.write(hyper_param)

    cfg_l, cfg_h, dy_cfg_f = get_cfg_range(cfg_scale)

    disc = XPDiscriminator(classifier_path) if (classifier_path is not None) and os.path.exists(classifier_path) else None

    if disc is not None:
        print('use convnext discriminator')

    unet = shared.sd_model.model.diffusion_model
    unet_down = unet.input_blocks
    unet_up = unet.output_blocks
    #,print(shared.sd_model.model.diffusion_model)

    def get_convs(block):
        return block[1].norm, block[1].proj_in, block[1].proj_out

    unet_part_list = [
        unet_down[1],
        unet_down[2][0], *get_convs(unet_down[2]), unet_down[3],
        unet_down[4][0], *get_convs(unet_down[4]), unet_down[5][0], *get_convs(unet_down[5]), unet_down[6],

        unet_up[8][0], *get_convs(unet_up[8]), unet_up[8][2], unet_up[7][0], *get_convs(unet_up[7]),
        unet_up[10][0], *get_convs(unet_up[10]), unet_up[9][0], *get_convs(unet_up[9]),
        unet_up[11],
    ]

    if unet_train:
        for layer in unet_part_list:
            layer.requires_grad_(True)
        unet.train()
    unet_lr = float(unet_lr)

    num_warmup_steps = 100
    num_training_steps = steps
    num_cycles = 0.5
    rate_min = 0.1
    def lr_lambda_cos(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        rate = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        return max(0.0, rate_min+rate*(1-rate_min))

    if use_negative:
        #optimizer = torch.optim.AdamW([embedding.vec, embedding_neg.vec], lr=scheduler.learn_rate)
        optimizer = torch.optim.AdamW([
                {'params': embedding.vec},
                {'params': embedding_neg.vec, 'lr': scheduler.learn_rate*neg_lr_w},
            ], lr=scheduler.learn_rate, betas=(adam_beta1, adam_beta2))
        optimizer_unet = torch.optim.AdamW([
            {'params': layer.parameters(), 'initial_lr': unet_lr} for layer in unet_part_list
        ], lr=unet_lr, eps=1e-6)
        #scheduler_unet = get_constant_schedule_with_warmup(optimizer_unet, num_warmup_steps=100, last_epoch=ititial_step)
        scheduler_unet = LambdaLR(optimizer_unet, lr_lambda_cos, ititial_step)
    else:
        optimizer = torch.optim.AdamW([embedding.vec], lr=scheduler.learn_rate)


    losses = torch.zeros((32,))

    last_saved_file = "<none>"
    last_saved_image = "<none>"
    forced_filename = "<none>"
    embedding_yet_to_be_embedded = False

    pbar = tqdm.tqdm(enumerate(ds), total=steps-ititial_step)
    for i, entries in pbar:
        log_here = f"Entry: {entries[0].cond_text} {entries[0].cond_text_neg}"
        shared.state.textinfo = log_here

        embedding.step = i + ititial_step
        if use_negative:
            embedding_neg.step = i + ititial_step

        scheduler.apply(optimizer, embedding.step)
        if scheduler.finished:
            break

        if shared.state.interrupted:
            break

        with torch.autocast("cuda"):
            #c = cond_model([entry.cond_text for entry in entries])
            if use_negative:
                #uc = cond_model([entry.cond_text_neg.replace(ds.placeholder_token, ds.placeholder_token+'-neg') for entry in entries])
                c_in = cond_model([entry.cond_text_neg.replace(ds.placeholder_token, ds.placeholder_token+'-neg') for entry in entries]+
                                  [entry.cond_text for entry in entries])
            else:
                c_in = cond_model([entry.cond_text for entry in entries])

            x = torch.stack([entry.latent for entry in entries]).to(devices.device)

            if use_att_map:
                att_mask = torch.stack([(entry.att_mask if entry.att_mask is not None else torch.ones_like(entry.latent)) for entry in entries]).to(devices.device)
                output = shared.sd_model(x, c_in, scale=(cfg_l, cfg_h), att_mask=att_mask, dy_cfg_f=dy_cfg_f)
            else:
                output = shared.sd_model(x, c_in, scale=(cfg_l, cfg_h), att_mask=None, dy_cfg_f=dy_cfg_f)

            if disc is not None or use_rec:
                if hasattr(shared.sd_model.decode_first_stage, '__wrapped__'):
                    x_samples_ddim = shared.sd_model.decode_first_stage.__wrapped__(shared.sd_model, output[2])  # forward with grad
                else:
                    x_samples_ddim = shared.sd_model.decode_first_stage(output[2])

            if disc is not None:
                # loss = ce(disc.get_all(x_samples_ddim), disc_label)
                loss = (1 - disc.get_score(x_samples_ddim)).mean()
            elif use_rec:
                loss = output[0] + F.l1_loss(torch.cat([entry.timg for entry in entries]), x_samples_ddim) * rec_loss_w
            else:
                loss = output[0]
            del x

            losses[embedding.step % losses.shape[0]] = loss.item()
            loss = loss / accumulation_steps

            if (i + 2) % accumulation_steps == 0:
                optimizer.zero_grad()
                if unet_train:
                    optimizer_unet.zero_grad()

            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                if unet_train:
                    for layer in unet_part_list:
                        torch.nn.utils.clip_grad_norm_(layer.parameters(), 1)
                    optimizer_unet.step()
                    scheduler_unet.step()

                torch.nn.utils.clip_grad_norm_(embedding.vec, 1)
                torch.nn.utils.clip_grad_norm_(embedding_neg.vec, 1)
                optimizer.step()

            with torch.no_grad():
                if ema_w != 1:
                    ema.update_model_average(embedding_ema, embedding)
                    if (i+1)%ema_rep_step == 0:
                        embedding.vec.data = deepcopy(embedding_ema.vec.data)

                if ema_w_neg != 1:
                    ema_neg.update_model_average(embedding_neg_ema, embedding_neg)
                    if (i + 1) % ema_rep_step_neg == 0:
                        embedding_neg.vec.data = deepcopy(embedding_neg_ema.vec.data)

        steps_done = embedding.step + 1

        ds_len = ds.batch_size
        epoch_num = embedding.step // ds_len
        epoch_step = embedding.step % ds_len

        pbar.set_description(f"[Epoch {epoch_num}: {epoch_step}/{ds_len}]loss: {losses.mean():.7f}, "
                             f"grad:{embedding.vec.grad.detach().cpu().abs().mean().item():.7f}, "
                             f"grad_neg:{embedding_neg.vec.grad.detach().cpu().abs().mean().item() if use_negative else 0:.7f}")

        if embedding_dir is not None and steps_done % save_embedding_every == 0:
            # Before saving, change name to match current checkpoint.
            embedding_name_every = f'{embedding_name}-{steps_done}'
            last_saved_file = os.path.join(embedding_dir, f'{embedding_name_every}.pt')
            save_embedding(embedding, checkpoint, embedding_name_every, last_saved_file, remove_cached_checksum=True,
                           use_negative=use_negative, embedding_neg=embedding_neg, unet_layers=unet_part_list)
            embedding_yet_to_be_embedded = True

        write_loss(log_directory, "prompt_tuning_loss.csv", embedding.step, ds_len, {
            "loss": f"{losses.mean():.7f}",
            "learn_rate": scheduler.learn_rate
        })

        if images_dir is not None and steps_done % create_image_every == 0:
            forced_filename = f'{embedding_name}-{steps_done}'
            last_saved_image = os.path.join(images_dir, forced_filename)

            shared.sd_model.first_stage_model.to(devices.device)

            p = processing.StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model,
                prompt=preview_prompt,
                do_not_save_grid=True,
                do_not_save_samples=True,
                do_not_reload_embeddings=True,
                negative_prompt=preview_prompt.replace(ds.placeholder_token, ds.placeholder_token + '-neg') if use_negative else None,
                cfg_scale=float(cfg_scale) if use_negative else 1.0,
            )

            if preview_from_txt2img:
                p.prompt = preview_prompt
                p.negative_prompt = preview_negative_prompt

                p.steps = preview_steps
                p.sampler_index = preview_sampler_index
                p.cfg_scale = preview_cfg_scale
                p.seed = preview_seed
                p.width = preview_width
                p.height = preview_height
            else:
                p.prompt = entries[0].cond_text
                if use_negative:
                    p.negative_prompt = entries[0].cond_text_neg.replace(ds.placeholder_token, ds.placeholder_token + '-neg')
                    p.cfg_scale = 7.0
                p.steps = 20
                p.width = training_width
                p.height = training_height

            preview_text = p.prompt

            processed = processing.process_images(p)
            image = processed.images[0]

            if unload:
                shared.sd_model.first_stage_model.to(devices.cpu)

            shared.state.current_image = image

            if save_image_with_stored_embedding and os.path.exists(last_saved_file) and embedding_yet_to_be_embedded:

                last_saved_image_chunks = os.path.join(images_embeds_dir, f'{embedding_name}-{steps_done}.png')

                info = PngImagePlugin.PngInfo()
                data = torch.load(last_saved_file)
                info.add_text("sd-ti-embedding", embedding_to_b64(data))

                title = "<{}>".format(data.get('name', '???'))

                try:
                    vectorSize = list(data['string_to_param'].values())[0].shape[0]
                except Exception as e:
                    vectorSize = '?'

                checkpoint = sd_models.select_checkpoint()
                footer_left = checkpoint.model_name
                footer_mid = '[{}]'.format(checkpoint.hash)
                footer_right = '{}v {}s'.format(vectorSize, steps_done)

                captioned_image = caption_image_overlay(image, title, footer_left, footer_mid, footer_right)
                captioned_image = insert_image_data_embed(captioned_image, data)

                captioned_image.save(last_saved_image_chunks, "PNG", pnginfo=info)
                embedding_yet_to_be_embedded = False

            last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt, shared.opts.samples_format, processed.infotexts[0], p=p, forced_filename=forced_filename, save_to_dirs=False)
            last_saved_image += f", prompt: {preview_text}"

            #set seed, seed is change by p
            seed+=1
            set_seed(seed)

        shared.state.job_no = embedding.step

        shared.state.textinfo = f"""
<p>
Loss: {losses.mean():.7f}<br/>
Step: {embedding.step} (Accumulation: {((i + 1) % accumulation_steps) + 1})<br/>
Last prompt: {html.escape(entries[0].cond_text)}<br/>
Last negative prompt: {html.escape(entries[0].cond_text_neg.replace(ds.placeholder_token, ds.placeholder_token+'-neg'))}<br/>
Last saved embedding: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""

    filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')
    save_embedding(embedding, checkpoint, embedding_name, filename, remove_cached_checksum=True, use_negative=use_negative, embedding_neg=embedding_neg,
                   unet_layers=unet_part_list)
    shared.sd_model.first_stage_model.to(devices.device)

    shared.sd_model.p_losses = p_losses_backup
    for layer in unet_part_list:
        layer.requires_grad_(True)
    unet.eval()

    return embedding, filename


def save_embedding(embedding, checkpoint, embedding_name, filename, remove_cached_checksum=True, use_negative=False, embedding_neg=None, unet_layers=None):
    old_embedding_name = embedding.name
    old_sd_checkpoint = embedding.sd_checkpoint if hasattr(embedding, "sd_checkpoint") else None
    old_sd_checkpoint_name = embedding.sd_checkpoint_name if hasattr(embedding, "sd_checkpoint_name") else None
    old_cached_checksum = embedding.cached_checksum if hasattr(embedding, "cached_checksum") else None
    try:
        embedding.sd_checkpoint = checkpoint.hash
        embedding.sd_checkpoint_name = checkpoint.model_name
        if remove_cached_checksum:
            embedding.cached_checksum = None
        embedding.name = embedding_name
        embedding.save(filename)

        if use_negative:
            embedding_neg.sd_checkpoint = checkpoint.hash
            embedding_neg.sd_checkpoint_name = checkpoint.model_name
            if remove_cached_checksum:
                embedding_neg.cached_checksum = None
            embedding_neg.name = embedding_name+'-neg'
            embedding_neg.save(f'{filename[:-3]}-neg.pt')

    except:
        embedding.sd_checkpoint = old_sd_checkpoint
        embedding.sd_checkpoint_name = old_sd_checkpoint_name
        embedding.name = old_embedding_name
        embedding.cached_checksum = old_cached_checksum

        if use_negative:
            embedding_neg.sd_checkpoint = old_sd_checkpoint
            embedding_neg.sd_checkpoint_name = old_sd_checkpoint_name
            embedding_neg.name = old_embedding_name+'-neg'
            embedding_neg.cached_checksum = old_cached_checksum

        raise

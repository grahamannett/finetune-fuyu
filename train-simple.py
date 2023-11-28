from dataclasses import dataclass
import random

import torch
import transformers
from simple_parsing import ArgumentParser, Serializable


@dataclass
class TrainConfig(Serializable):
    # since slurm seems to fuck up progress bar (so cant see in wandb/log.o%job)
    batch_log_every: int = False  # log
    num_iters: int = False  # num iters if not going through full dataset

    model_name: str = "adept/fuyu-8b"

    epochs: int = 10
    grad_accum_steps: int = 4

    dl_batch_size: int = 2
    dl_num_workers: int = 0
    dl_pin_memory: bool = True

    weight_decay: float = 0.0
    gradient_clipping: float = 1.0
    learning_rate: float = 1e-03
    scheduler_type: str = "cosine"
    gamma: float = 0.85


@dataclass
class DataCollator:
    processor: transformers.models.fuyu.FuyuProcessor

    def __call__(self, samples: list[dict]):
        text = [sample["text"] for sample in samples]
        images = [sample["image"] for sample in samples]
        return self.processor(text=text, images=images, return_tensors="pt")


class MockDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        words_file: str = "/usr/share/dict/words",
        max_words: int = 100,
        num_iters: int = 100,
        image_max: tuple[int, int] = [1000, 1000],
        image_min: tuple[int, int] = [500, 500],
    ):
        self.num_iters = num_iters
        self.max_words = max_words
        self.image_min, self.image_max = image_min, image_max

        with open(words_file, "r") as f:
            self.words = [word.strip() for word in f.readlines()]

    def __len__(self):
        return self.num_iters

    def __getitem__(self, idx):
        return self.generate()

    def generate(self):
        num_words = random.randint(1, self.max_words)
        text = " ".join([random.choice(self.words) for _ in range(num_words)])
        image_dim = [
            random.randint(self.image_min[0], self.image_max[0]),
            random.randint(self.image_min[1], self.image_max[1]),
        ]
        image = torch.rand(3, *image_dim)
        return {"text": text, "image": image}


def loss_fn(logits, labels):
    # b, l needed when fsdp
    b, l, c = logits.shape

    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, c)
    shift_labels = shift_labels.view(-1)

    shift_labels = shift_labels.to(shift_logits.device)
    loss = torch.nn.functional.cross_entropy(shift_logits.float(), shift_labels)
    return loss


def train(
    train_config: TrainConfig,
    model,
    train_dataloader,
    optimizer,
    scheduler,
):
    model.train()
    for epoch in range(train_config.epochs):
        # reset loss per epoch
        losses = 0

        for batch_idx, batch in enumerate(train_dataloader):
            batch["input_ids"] = batch["input_ids"].to(model.device)
            batch["image_patches"] = [im.to(model.device) for im in batch["image_patches"]]
            batch["attention_mask"] = batch["attention_mask"].to(model.device)
            batch["image_patches_indices"] = batch["image_patches_indices"].to(model.device)

            outputs = model(**batch)

            loss = loss_fn(outputs.logits, batch["input_ids"])
            loss = loss / train_config.grad_accum_steps

            loss.backward()

            if train_config.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping)

            if (batch_idx % train_config.grad_accum_steps) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            losses += loss.item()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(TrainConfig, dest="train_config")
    args = parser.parse_args()

    train_config: TrainConfig = args.train_config

    processor = transformers.models.fuyu.FuyuProcessor.from_pretrained(train_config.model_name)
    model = transformers.models.fuyu.FuyuForCausalLM.from_pretrained(train_config.model_name, device_map="auto")

    train_dataset = MockDataset()
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=DataCollator(processor),
        batch_size=train_config.dl_batch_size,
        num_workers=train_config.dl_num_workers,
        pin_memory=train_config.dl_pin_memory,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    train(
        train_config,
        model,
        train_dl,
        optimizer=optimizer,
        scheduler=scheduler,
    )

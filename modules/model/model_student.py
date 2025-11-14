from torch import optim, nn, utils, Tensor
import lightning as L
import torch
import torch.nn


class LitAutoEncoder_student(L.LightningModule):
    def __init__(self, num_inputs, teacher_model, temperature, comp_coeff=9, light_dimension=2, num_outputs=3,
                 alpha=0.1):
        super().__init__()

        self.num_inputs = num_inputs  # number of input, for RTI will be most probably 3 times the number of source images
        self.comp_coeff = comp_coeff  # number of computed coefficients, per pixel, in the latent space
        self.light_dimension = light_dimension  # number of light dimensions for RTI: 2 -> (x,y) | 3 -> (x,y,z)
        self.num_outputs = num_outputs  # number of outputs from the decoder, for RTI is commonly 3 (RGB channels)
        self.encoder, self.decoder = self.autoencoder()
        self.teacher_decoder = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.teacher_decoder = torch.load((teacher_model + "/decoder.pth")).to(device)    
        self.teacher_encoder = torch.load((teacher_model + "/encoder.pth")).to(device)
        for param in self.teacher_decoder.parameters():
            param.requires_grad = False
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss = self.common_step_distillation(batch, batch_idx)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self.common_step_distillation(batch, batch_idx)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx, dataloader_idx=0):

        loss = self.common_step_distillation(batch, batch_idx)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
        return {"loss": loss}

    def predict_step(self, batch, batch_idx):
        loss = self.common_step_distillation(batch, batch_idx)
        self.log("pred_loss", loss)
        return {"loss": loss}

    def common_step_distillation(self, batch, batch_idx):
        x, y, g = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        zy = torch.cat((z, y), dim=1)
        student_pred = self.decoder(zy)
        student_loss = nn.functional.mse_loss(student_pred, g)

        z_teacher = self.teacher_encoder(x)
        zy_teacher = torch.cat((z_teacher, y), dim=1)
        teach_pred = self.teacher_decoder(zy_teacher)
        distillation_loss = nn.functional.mse_loss(student_pred / self.temperature, teach_pred / self.temperature)
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        del x, y, g, z, zy, z_teacher, zy_teacher, teach_pred, student_loss, student_pred, distillation_loss
        torch.cuda.empty_cache()
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-2, amsgrad=True)
        return optimizer

    def on_train_epoch_end(self):
        epoch = self.current_epoch  # Get current epoch number
        print(f"\nTraining completed for epoch {epoch}")  # Print to console

    def on_train_end(self, ):
        print("Training is done.")

    #     # do something with all preds
    # self.training_step_outputs.clear()  # free memory
    def autoencoder(self):
        encoder = nn.Sequential(nn.Linear(self.num_inputs, 150),
                                nn.ELU(),
                                nn.Linear(150, 150),
                                nn.ELU(),
                                nn.Linear(150, 150),
                                nn.ELU(),
                                nn.Linear(150, 150),
                                nn.ELU(),
                                nn.Linear(150, self.comp_coeff))

        decoder = nn.Sequential(nn.Linear(self.comp_coeff + self.light_dimension, 20),
                                nn.ELU(),
                                nn.Linear(20, 20),
                                nn.ELU(),
                                nn.Linear(20, 3))
        return encoder, decoder

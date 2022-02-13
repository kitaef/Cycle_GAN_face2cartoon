import torch

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_dl = DataLoader(test_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

def train(epoch_start=0, epoch_end=5):
    epochs = epoch_end

    for epoch in range(epoch_start, epoch_end):
        progress_bar = tqdm(enumerate(train_dl), total=len(train_dl))
        for i, data in progress_bar:
            G_A2B.train()
            G_B2A.train()
            D_A.train()
            D_B.train()
            real_a = data[0].to(device)
            real_b = data[1].to(device)
            batch_size = real_a.size(0)
            # create labels for real and fake images.
            real_label = torch.full((batch_size, 1), 1, device=device, dtype=torch.float32).uniform_(0.8, 1.0)
            fake_label = torch.full((batch_size, 1), 0, device=device, dtype=torch.float32).uniform_(0., 0.2)

            #### training Generators A2B and B2A:
            optim_G_A2B.zero_grad()
            optim_G_B2A.zero_grad()
            # Identity loss
            # G_B2A(A) should equal A if real A is fed
            transformed_a2a = G_B2A(real_a)
            loss_identity_a = identity_loss(transformed_a2a, real_a) * 1
            # G_A2B(B) should equal B if real B is fed
            transformed_b2b = G_A2B(real_b)
            loss_identity_b = identity_loss(transformed_b2b, real_b) * 1
            # GAN loss D_A(G_A(A))
            fake_a = G_B2A(real_b)
            fake_a_pred = D_A(fake_a)
            loss_GAN_B2A = adversarial_loss(fake_a_pred, real_label)
            # GAN loss D_B(G_B(B))
            fake_b = G_A2B(real_a)
            fake_b_pred = D_B(fake_b)
            loss_GAN_A2B = adversarial_loss(fake_b_pred, real_label)
            # Cycle loss
            recovered_b2a = G_B2A(fake_b)
            loss_cycle_ABA = cycle_loss(recovered_b2a, real_a) * 3
            recovered_a2b = G_A2B(fake_a)
            loss_cycle_BAB = cycle_loss(recovered_a2b, real_b) * 3
            # Combine loss and calculate gradients
            loss_G_A2B = loss_GAN_A2B + loss_cycle_BAB + loss_identity_b
            loss_G_B2A = loss_GAN_B2A + loss_cycle_ABA + loss_identity_a
            if (i + 1) % logging_step == 0:
                losses_G_A2B.append(loss_G_A2B.item())
                losses_G_B2A.append(loss_G_B2A.item())
            # Update G_A and G_B's weights
            loss_G_A2B.backward(retain_graph=True)
            loss_G_B2A.backward()
            optim_G_A2B.step()
            optim_G_B2A.step()
            if accelerator == 'tpu':  # for training on 1-core of TPU
                xm.mark_step()

            #### training Discriminator A
            optim_D_A.zero_grad()
            # Real A image loss
            real_a_pred = D_A(real_a)
            loss_D_real_a = adversarial_loss(real_a_pred, real_label)
            # Fake A image loss
            fake_a = fake_A_buffer.push_and_pop(fake_a)
            fake_a_pred = D_A(fake_a.detach())
            loss_D_fake_a = adversarial_loss(fake_a_pred, fake_label)
            # Combined loss and calculate gradients
            loss_D_A = (loss_D_real_a + loss_D_fake_a)
            if (i + 1) % logging_step == 0:
                losses_D_A.append(loss_D_A.item())
            # Calculate gradients for D_A
            loss_D_A.backward()
            # Update D_A weights
            optim_D_A.step()
            if accelerator == 'tpu':
                xm.mark_step()

            #### training Discriminator B
            optim_D_B.zero_grad()
            # Real B image loss
            real_b_pred = D_B(real_b)
            loss_D_real_b = adversarial_loss(real_b_pred, real_label)
            # Fake B image loss
            fake_b = fake_B_buffer.push_and_pop(fake_b)
            fake_b_pred = D_B(fake_b.detach())
            loss_D_fake_b = adversarial_loss(fake_b_pred, fake_label)
            # Combined loss and calculate gradients
            loss_D_B = (loss_D_real_b + loss_D_fake_b)
            if (i + 1) % logging_step == 0:
                losses_D_B.append(loss_D_B.item())
            # Calculate gradients for D_B
            loss_D_B.backward()
            # Update D_B weights
            optim_D_B.step()
            if accelerator == 'tpu':
                xm.mark_step()

            progress_bar.set_description(
                f"[{epoch + 1}/{epochs}][{i + 1}/{len(train_dl)}] "
                f"Loss_D_A: {(loss_D_A).item():.4f} "
                f"Loss_D_B: {(loss_D_B).item():.4f} "
                f"Loss_G_A2B: {loss_G_A2B.item():.4f} "
                f"Loss_G_B2A: {loss_G_B2A.item():.4f} "
                f"Loss_G_identity: {(loss_identity_a + loss_identity_b).item():.4f} "
                f"loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.4f} "
                f"loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB).item():.4f}")
            #     clear_output()
            # save models
            if (i + 1) % saving_step == 0:
                model_path = f'/content/drive/MyDrive/DLS FINAL/13_02/new_{epoch + 1}_{i + 1}_'
                torch.save(G_A2B.cpu().state_dict(), model_path + 'G_A2B')
                torch.save(G_B2A.cpu().state_dict(), model_path + 'G_B2A')
                torch.save(D_A.cpu().state_dict(), model_path + 'D_A')
                torch.save(D_B.cpu().state_dict(), model_path + 'D_B')
                G_A2B.to(device)
                G_B2A.to(device)
                D_A.to(device)
                D_B.to(device)

            # show results
            if (i + 1) % logging_step == 0:
                G_A2B.eval()
                G_B2A.eval()
                D_A.eval()
                D_B.eval()
                test_iter = iter(train_dl)
                rows = 5
                a_real, b_real = next(test_iter)
                if BATCH_SIZE < 4:
                    for _ in range(rows - 1):
                        next_a, next_b = next(test_iter)
                        a_real = torch.vstack((a_real, next_a))
                        b_real = torch.vstack((b_real, next_b))
                a_real.to(device)
                b_real.to(device)
                plt.figure(figsize=(18, rows + 3))
                grid = plt.GridSpec(rows + 5, 3, wspace=.1, hspace=.01)
                with torch.no_grad():
                    a_transformed = G_A2B(a_real.to(device))
                    b_transformed = G_B2A(b_real.to(device))
                    a_recovered = G_B2A(a_transformed)
                    b_recovered = G_A2B(b_transformed)
                show(a_real, a_transformed, a_recovered,
                     b_real, b_transformed, b_recovered,
                     grid, rows=rows)
                plt.subplot(grid[rows:rows + 5, 0:3])
                plt.plot(losses_G_A2B, label="G_A2B")
                plt.plot(losses_G_B2A, label="G_B2A")
                plt.plot(losses_D_A, label="D_A")
                plt.plot(losses_D_B, label="D_B")
                plt.legend()
                plt.suptitle(t=(
                    f"[Epoch {epoch + 1}/{epochs}], Batch[{i + 1}/{len(train_dl)}]\n"
                    f"Loss_D_A: {(loss_D_A).item():.4f}\n"
                    f"Loss_D_B: {(loss_D_B).item():.4f}\n"
                    f"Loss_G_A2B: {loss_G_A2B.item():.4f}\n"
                    f"Loss_G_B2A: {loss_G_B2A.item():.4f}\n"
                    f"Loss_G_identity: {(loss_identity_a + loss_identity_b).item():.4f}\n"
                    f"loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.4f}\n"
                    f"loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB).item():.4f}"
                ), y=0.8, x=0.75
                )
                plt.show()
                # Update learning rates
    #     lr_scheduler_G.step()
    #     lr_scheduler_D_A.step()
    #     lr_scheduler_D_B.step()


print(f'logging step: {logging_step}')
print(f'saving step: {saving_step}')
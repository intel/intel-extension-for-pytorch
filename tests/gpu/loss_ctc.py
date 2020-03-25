import torch
import torch_ipex

def print_detial_dif(a, b, rtol=1e-05, atol=1e-08):
    print(a)
    print(b)
    comp = torch.abs(a - b)
    margin = atol + rtol * torch.abs(b)
    mask = comp.gt(margin)
    print("a", a[mask].detach().numpy())
    print("b", b[mask].detach().numpy())
    print("diff", comp[mask].detach().numpy())
    print("marging", margin[mask].detach().numpy())
    print("relative diff ratio", (comp[mask]/b[mask]).detach().numpy())

def test_loss_ctc(log_probs, targets, input_lengths, target_lengths):
    log_probs.requires_grad_(True)
    loss = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)
    print("loss", loss)
    loss.backward()

    log_probs.requires_grad_(False)
    log_probs_dpcpp = log_probs.to("dpcpp")
    log_probs_dpcpp.requires_grad_(True)
    targets_dpcpp = targets.to("dpcpp")
    input_lengths_dpcpp = input_lengths.to("dpcpp")
    target_lengths_dpcpp = target_lengths.to("dpcpp")

    loss_dpcpp = torch.nn.functional.ctc_loss(log_probs_dpcpp, targets_dpcpp, input_lengths_dpcpp, target_lengths_dpcpp)
    print("loss_dpcpp", loss_dpcpp.cpu())

    loss_dpcpp.backward()

    cpu_grad = log_probs.grad;
    dpcpp_grad = log_probs_dpcpp.grad.cpu()
    # all close
    if not torch.allclose(cpu_grad, dpcpp_grad, atol=1e-05):
        printf("allclose fail")
        # print_detial_dif(cpu_grad, dpcpp_grad)
        # raise Exception("{} backward error".format("ctc_loss"))


log_probs = torch.randn(50, 15, 20).log_softmax(2).detach().requires_grad_()
targets = torch.randint(1, 20, (15, 30), dtype=torch.long)
input_lengths = torch.full((15,), 50, dtype=torch.long)
target_lengths = torch.randint(10, 30, (15,), dtype=torch.long)

test_loss_ctc(log_probs, targets, input_lengths, target_lengths)

log_probs = torch.randn(50, 17, 20).log_softmax(2).detach().requires_grad_()
targets = torch.randint(1, 20,
                        (17, 30), dtype=torch.long)
input_lengths = torch.full((17,), 50, dtype=torch.long)
target_lengths = torch.randint(10, 30, (17,), dtype=torch.long)

test_loss_ctc(log_probs, targets, input_lengths, target_lengths)

log_probs = torch.randn(250, 16, 20).log_softmax(2).detach().requires_grad_()
targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
input_lengths = torch.full((16,), 250, dtype=torch.long)
target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)

test_loss_ctc(log_probs, targets, input_lengths, target_lengths)


print("pass")

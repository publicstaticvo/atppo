import torch
try:
    from apex import amp
except ImportError:
    print("ImportError: no modul named amp")


def step(loss, model, args, optimizer=None, scheduler=None):
    if args.ds_config:
        model.backward(loss)
        model.step()
    else:
        if args.apex_level > 0:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if args.grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.grad_norm)
        else:
            loss.backward()
            if args.grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        if model.step_count % args.grad_acc == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
import torch
try:
    from apex import amp
except ImportError:
    print("ImportError: no modul named amp")


def concat_audio(audio, transcript):
    new_audio = []
    for i, t in enumerate(transcript):
        if i > 0:
            new_audio.append(torch.zeros(1600, dtype=audio.dtype))
        new_audio.append(audio[t[0]:t[1]])
    new_audio = torch.cat(new_audio)
    return new_audio


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
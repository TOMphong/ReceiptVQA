import torch
from Model import Transformer
from Data.vocab import Vocab
from Utils import tokenizing


def greedy_decode(input_text="", 
                  model=None, 
                  src_vocab=None, 
                  tgt_vocab=None, 
                  max_output_length=50,
                  device="cuda"):
    assert model!=None, "Need trained model"
    assert src_vocab!=None, "Need source vocab"
    assert tgt_vocab!=None, "Need target vocab"

    input_tokens = [src_vocab[i] for i in tokenizing(input_text)]
    
    encoder_input = torch.Tensor([input_tokens]).to(device)
    model.eval()
    with torch.no_grad():
        encoder_output = model.encode(encoder_input)

    decoder_input = torch.LongTensor([[tgt_vocab.BOS]]).to(device)
    # Autoregressive
    for _ in range(max_output_length):
        # Decoder prediction
        logits = model.decode(encoder_output, decoder_input)

        # Greedy selection
        token_index = torch.argmax(logits[:, -1], keepdim=True)
        
        # EOS is most probable => Exit
        if token_index.item()==tgt_vocab.EOS:
            break

        # Next Input to Decoder
        decoder_input = torch.cat([decoder_input, token_index], dim=1)
    
    decoder_output = decoder_input[0, 1:]#.numpy()

    return [tgt_vocab[o.item()] for o in decoder_output]

def sequence_length_penalty(length: int, alpha: float=0.6) -> float:
    return ((5 + length) / (5 + 1)) ** alpha


def beam_decode(input_text="", 
                model=None, 
                src_vocab=None, 
                tgt_vocab=None,
                beam_size = 3, 
                max_output_length=50,
                alpha = 0.6):
    assert model!=None, "Need trained model"
    assert src_vocab!=None, "Need source vocab"
    assert tgt_vocab!=None, "Need target vocab"

    input_tokens = [src_vocab[i] for i in tokenizing(input_text)]
    encoder_input = torch.Tensor([input_tokens])
    model.eval()
    with torch.no_grad():
        encoder_output = model.encode(encoder_input)

    decoder_input = torch.Tensor([[tgt_vocab.BOS]]).long()
    scores = torch.Tensor([0.])
    vocab_size = len(tgt_vocab)

    # Autoregressive
    for i in range(max_output_length):
        # Decoder prediction
        logits = model.decode(encoder_output, decoder_input)

        log_probs = torch.log_softmax(logits[:,-1], dim=1)
        log_probs = log_probs / sequence_length_penalty(i+1, alpha)

        log_probs[decoder_input[:, -1]==tgt_vocab.EOS, :] = 0

        scores = scores.unsqueeze(1) + log_probs

        scores, indices = torch.topk(scores.reshape(-1), beam_size)
        beam_indices = torch.div(indices, vocab_size, rounding_mode='float')

        token_indices = torch.remainder(indices, vocab_size)

        next_decoder_input = []
        for beam_index, token_index in zip(beam_indices, token_indices):
            prev_decoder_input = decoder_input[beam_index]
            if prev_decoder_input[-1] == tgt_vocab.EOS:
                token_index = tgt_vocab.EOS
            token_index = torch.LongTensor([token_index])
            next_decoder_input.append(torch.cat([prev_decoder_input, token_index]))
        decoder_input = torch.vstack(next_decoder_input)

        if(decoder_input[:, -1]==tgt_vocab.EOS).sum()==beam_size:
            break

        if i==0:
            encoder_output = encoder_output.expand(beam_size,
                                                   *encoder_output.shape[1:])
    decoder_output, _ = max(zip(decoder_input, scores), key=lambda x: x[1])
    decoder_output = decoder_output[1:].numpy()

    output = [tgt_vocab[i] for i in decoder_output if i != tgt_vocab.EOS]
    
    return output
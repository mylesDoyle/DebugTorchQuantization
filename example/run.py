import torch

backend = "fbgemm"
# backend = "qnnpack"
torch.backends.quantized.engine = backend

from example.model import FooConv1x1

torch.manual_seed(0)

# Hard code relevant quantization parameters
def set_qconfig_params(model_prepared, k):
    # Conv weight
    model_prepared.conv.weight_fake_quant.scale = torch.Tensor([2.0*k/255.0])   # Symmetric, hence multiply by 2
    model_prepared.conv.weight_fake_quant.activation_post_process.min_val = torch.tensor(0.0)
    model_prepared.conv.weight_fake_quant.activation_post_process.max_val = torch.tensor(k)

    # Requantization
    model_prepared.conv.activation_post_process.scale = torch.Tensor([k/255.0])
    model_prepared.conv.activation_post_process.min_val = torch.tensor(0.0)
    model_prepared.conv.activation_post_process.max_val = torch.tensor(k)
    model_prepared.conv.activation_post_process.activation_post_process.min_val = torch.tensor(0.0)
    model_prepared.conv.activation_post_process.activation_post_process.max_val = torch.tensor(k)

    # Input quant stub
    model_prepared.quant.activation_post_process.scale = torch.Tensor([1.0/255.0])
    model_prepared.quant.activation_post_process.activation_post_process.min_val = torch.tensor(0.0)
    model_prepared.quant.activation_post_process.activation_post_process.max_val = torch.tensor(1.0)

if __name__ == "__main__":
   
    input_fp32 = torch.arange(0,256).repeat(1,3,256,1)/255.0   # 0 to 255 repeated across rows, then normalized to [0,1]

    model = FooConv1x1(set_qconfig=True)    # Prepare model with QConfig defined
    k = 1.0 # Set Conv layer multiplier
    model.set_weights(k)    # Set bias to zero and conv weights to k*Identity
    model.fuse()    # fuse conv and ReLU 

    model_prepared = torch.quantization.prepare_qat(model).train()  # prepare_qat required to set weight qparams
    model_prepared.eval()

    model_prepared.apply(torch.quantization.disable_observer).eval()    # Disable QConfig Observers

    set_qconfig_params(model_prepared, k)   # Set quantization parameters to theoretical values

    expected_output_fp32 = model_prepared(input_fp32)
    expected_output_quint8 = (expected_output_fp32*(k*255)).to(torch.uint8)
    
    model_prepared.dequant = torch.nn.Identity()    # Disable the output dequant stub

    # Convert model so that it runs as fully quantized model
    model_quant = torch.quantization.convert(model_prepared, inplace=False)

    output_quint8_fp32 = model_quant(input_fp32) # fp32 outputs with scale and shift parameterising it to quint8

    error = torch.abs(expected_output_fp32 - output_quint8_fp32.dequantize())
    error_mean = torch.mean(error)
    error_max = torch.max(error)
    first_nonzero_index = error.nonzero()[0].tolist()

    print(f"{error_mean=}")
    print(f"{error_max=}")
    print(f"First nonzero: index: ({first_nonzero_index}")
    print(f"\tvalue fp32: {error[*first_nonzero_index]}")
    print(f"\tvalue expected quint8: {expected_output_quint8[*first_nonzero_index]}")
    print(f"\tvalue outputed quint8: {output_quint8_fp32.int_repr()[*first_nonzero_index]}")

    # import ipdb; ipdb.set_trace()
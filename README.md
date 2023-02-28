# DebugTorchQuantization
This repo is designed to test and highlight an observed case where a quantized model in Torch does not produce the expected results;
- Torch produces the expected result for a model prepared with `torch.quantization.prepare_qat`
- Torch produces unexpected results when the previously prepared model is converted to a fully quantized model with `torch.quantization.convert`

## Experiment

This repo was created to highlight a discrepancy in Torch when executing quantized (and fake quantized) models under different setup conditions;
- inferring with a normal QAT model (fake quantized) - produces expected results
- inferring with a prepared and converted model to int8 (quantized) - produces unexpected results

To highlight the issue, we set up a simple toy example as follows;

### Model

A simple Conv-ReLU fused model is defined with 
- bias set to zero
- conv weights set to `k*I` where `k` is some floating point scalar multiplier and `I` represents an identity matrix of the correct shape for the conv layer
- A quantization stub which quantizes the `fp32` inputs to `quint8`
- A dequantization stub which dequantizes the `quint8` outputs to `fp32` - note, this stub gets set to the identity for the fully int8 quantized model

### Inputs

Inputs are provided to the model in single precision floating point units in all cases. To highlight the issue, we consider passing a range of input values between 0 and 255 across an input image of size `[1,3,256,256]` and scaling the values to between 0 and 1 by dividing by 255. 

## Dependencies

The example in this repo was tested using
- Python 3.11.0
- Python packages installed with pip which are listed in the `requirements.txt` in the [repo provided](https://github.com/mylesDoyle/DebugTorchQuantization/blob/main/requirements.txt)

## To execute

Simply execute

```bash
python3 -m example.run
```

to run the application.

Interactive debugging can easily be enabled by uncommenting out the `ipdb` debugger line at the end of `example/run.py` for further testing.

## Observations

For simplicity, we compare the quantized outputs, but the same can be observed for the dequantized outputs.

- For the output of a model (fake or real quantized), we expect each row to be identical across all rows and channels. This was observed in all cases indicating determinism within a model's execution
- The quantized outputs of the (fake quantized) model prepared with `torch.quantization.prepare_qat` were as expected;
    - values ranging from 0 to 255, indicating a unique bin for each of the outputs which get dequantized into the expected output value
    - a summary of the first row of the first channel depicts the beginning, middle and end of that row
```
tensor([  0,   1, ..., 126, 127, 128, 129, 130, 131, ..., 253, 254, 255], dtype=torch.uint8)
```

- The quantized outputs of the (quantized) model converted with `torch.quantization.convert` were not quite as expected; 
    - values ranging from 0 to 254, implying we are losing information somewhere within the quantized execution 
    - a summary of the first row of the first channel depicts the beginning, middle and end of that row
```
tensor([  0,   1, ..., 126, 127, 127, 128, 129, 130, ..., 252, 253, 254], dtype=torch.uint8)
```
- Comparing the quantized outputs of the two models, we observe in the real quantized model;
    - a discrepancy can be seen as the value 127 is repeated twice, and all other values are shifted after that
    - this results in the repeated 127 value being the incorrect bin for its expected dequantized value, and all other values following this duplication have been shifted to incorrect bins as well
    - this behaviour is unexpected and results in non-determinism across both model's execution
    - note, it is interesting that this discrepancy appears for `128=255/2+1` and all following values due to 128 being halway bin of the possible range of bins 

## Conclusion

One of the main reasons for using quantization is to ensure determinism across different compute platforms, so the non-deterministic behaviour between a fake and real quantized model is extremely problematic, especially when it comes to deploying quantized models. It is clear from this example that the real quantized model is not working as expected. This must either be due to an error I have in my implementation or a bug within PyTorch. 

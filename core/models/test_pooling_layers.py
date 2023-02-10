
import torch


# Create a sample input tensor
input_tensor = torch.randn(10, 5, 7, 7)

# Create an instance of the BottomPool layer
bottom_pool = BottomPool(7, 7)

# Pass the input tensor through the layer
output_tensor = bottom_pool(input_tensor)

# The shape of the output tensor should be (10, 5, 1, 1)
print(output_tensor.shape)

# the output should be the same as the BottomPoolTorch layer
bottom_pool_torch = BottomPoolTorch()
output_tensor_torch = bottom_pool_torch(input_tensor)
print(torch.allclose(output_tensor, output_tensor_torch))
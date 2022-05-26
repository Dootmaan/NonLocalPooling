# Non-Local Pooling

## Update 2.0.1

- Now NonLocalPool2d will not use PixelShuffle by default since it may harm the performance. NonLocalPool3d still uses PixelShuffle3d to reduce the token number.

## Update 1.2.0

- Now you can determine the output channel just like all the other learnable pooling methods (former version force out_channel=in_channel). However, if you specify out_channel, MaxPool would not work then since they cannot be added together. Leave out_channel to None to make the module works like before.

---

To use NonLocalPooling for your PyTorch project:

## Step 1
```
pip install nonlocalpooling
```

## Step 2
```
from nonlocalpooling.pool import NonLocalPool2d, NonLocalPool3d
```

---

Non-Local Pooling can be used to substitue your original PyTorch pooling methods.

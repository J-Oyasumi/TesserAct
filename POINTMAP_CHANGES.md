# TesserActPointmap 类修改说明

## 需要修改的文件
`tesseract/modules/tesseract_model.py` 中的 `TesserActPointmap.forward` 方法

## 修改详情

### 1. 第 651-653 行 - Patch embedding 部分

**原代码:**
```python
# 2. Patch embedding
depth_input_state = hidden_states[:, :, channels // 3 : channels // 3 * 2]
normal_input_state = hidden_states[:, :, channels // 3 * 2 :]
```

**修改为:**
```python
# 2. Patch embedding
# For Pointmap: input channels are [RGB (16), Pointmap (16)] = 32 total
# Split: RGB is first half, Pointmap is second half
pointmap_input_state = hidden_states[:, :, channels // 2:]  # [B, T, 16, H, W]
```

**原因:**
- 原来处理 3 个模态 (RGB, Depth, Normal),每个占 channels//3
- 现在只处理 2 个模态 (RGB, Pointmap),每个占 channels//2

---

### 2. 第 714 行 - 注释修改

**原代码:**
```python
# rgb_output [b, t, c, h, w], depth_input_state [b, t, c * 2, h, w]
```

**修改为:**
```python
# rgb_output [b, t, c, h, w], pointmap_input_state [b, t, c, h, w]
```

---

### 3. 第 715-720 行 - 输入拼接和处理

**原代码:**
```python
rdn_input_state = torch.cat([rgb_output, depth_input_state, normal_input_state], dim=2).transpose(1, 2)
rdn_input_state = self.dn_out_proj(rdn_input_state).transpose(1, 2)
rdn_input_state = rdn_input_state.reshape(batch_size, num_frames, -1, height // p, p, width // p, p)
rdn_input_state = rdn_input_state.permute(0, 1, 3, 5, 2, 4, 6)  # [b, t, h/p, w/p, c, p, p]
rdn_input_state = rdn_input_state.flatten(4, 6).flatten(1, 3)  # [b, (t * h/p * w/p), c * p * p]
depth_states = self.dn_out(torch.cat([hidden_states, rdn_input_state], dim=-1))
```

**修改为:**
```python
# Concatenate RGB output and Pointmap input for prediction
rp_input_state = torch.cat([rgb_output, pointmap_input_state], dim=2).transpose(1, 2)
rp_input_state = self.dn_out_proj(rp_input_state).transpose(1, 2)
rp_input_state = rp_input_state.reshape(batch_size, num_frames, -1, height // p, p, width // p, p)
rp_input_state = rp_input_state.permute(0, 1, 3, 5, 2, 4, 6)  # [b, t, h/p, w/p, c, p, p]
rp_input_state = rp_input_state.flatten(4, 6).flatten(1, 3)  # [b, (t * h/p * w/p), c * p * p]
pointmap_states = self.dn_out(torch.cat([hidden_states, rp_input_state], dim=-1))
```

**原因:**
- 不再拼接 depth 和 normal,只拼接 pointmap
- 变量名从 `rdn_` (RGB-Depth-Normal) 改为 `rp_` (RGB-Pointmap)
- 输出变量从 `depth_states` 改为 `pointmap_states`

---

### 4. 第 722-730 行 - Unpatchify 部分

**原代码:**
```python
# Unpatchify depth states
if p_t is None:
    dn_output = depth_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
    dn_output = dn_output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
else:
    dn_output = depth_states.reshape(
        batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
    )
    dn_output = dn_output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)
```

**修改为:**
```python
# Unpatchify pointmap states
if p_t is None:
    pointmap_output = pointmap_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
    pointmap_output = pointmap_output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
else:
    pointmap_output = pointmap_states.reshape(
        batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
    )
    pointmap_output = pointmap_output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)
```

---

### 5. 第 732-733 行 - 最终输出

**原代码:**
```python
# 5. Output
output = torch.cat([rgb_output, dn_output], dim=2)
```

**修改为:**
```python
# 5. Output: concatenate RGB and Pointmap
output = torch.cat([rgb_output, pointmap_output], dim=2)
```

---

## 输入/输出 Shape 总结

### 输入:
- `hidden_states`: `[B, T, 32, H, W]`
  - RGB: `[B, T, 16, H, W]` (前 16 通道)
  - Pointmap: `[B, T, 16, H, W]` (后 16 通道)

### 输出:
- `output`: `[B, T, 32, H, W]`
  - RGB: `[B, T, 16, H, W]` (预测的 RGB)
  - Pointmap: `[B, T, 16, H, W]` (预测的 Pointmap)

---

## 对应的 Dataset 修改

确保 `RoboPointmap` dataset 返回的数据格式为 `[T, 6, H, W]`:
- RGB: 前 3 通道 (归一化到 [-1, 1])
- Pointmap XYZ: 后 3 通道 (归一化到 [-1, 1])

这会经过 VAE 编码后变成 `[T, 32, H//8, W//8]` 送入模型。

import spectral
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt


# ========== 1. 数据加载与预处理 ==========

def load_and_preprocess_data(img_file, hdr_file):
    # 使用 spectral 库打开高光谱数据
    crism_data = spectral.open_image(hdr_file).load()  # 加载数据为 NumPy 数组
    print(f"Data shape: {crism_data.shape}")  # (H, W, L)，高、宽、光谱波段数

    # 标准化到零均值单位方差
    mean = np.mean(crism_data, axis=(0, 1), keepdims=True)
    std = np.std(crism_data, axis=(0, 1), keepdims=True) + 1e-8
    crism_data = (crism_data - mean) / std
    print(f"Normalized data with mean ~0 and std ~1")

    H, W, L = crism_data.shape
    data_2d = crism_data.reshape(-1, L).astype(np.float32)  # 转换为浮点型
    data_3d = data_2d.reshape(-1, L, 1)  # 转换为 (N, L, 1) 格式适配 Conv1D
    print(f"Flattened data shape: {data_2d.shape}")  # (N, L)

    return data_3d, H, W, L, mean, std


# ========== 2. 构建生成器 (Autoencoder) ==========

def build_generator(input_dim):
    inp = layers.Input(shape=(input_dim, 1), name='gen_input')

    # Encoder
    x = layers.Conv1D(16, kernel_size=7, strides=2, padding='same')(inp)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(32, kernel_size=7, strides=2, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.BatchNormalization()(x)

    # Bottleneck
    shape_before_fc = tf.keras.backend.int_shape(x)  # (None, L/8, 64)
    x = layers.Flatten()(x)
    x = layers.Dense(shape_before_fc[1] * 64, activation='relu')(x)

    # Decoder
    x = layers.Dense(shape_before_fc[1] * 64, activation='relu')(x)
    x = layers.Reshape((shape_before_fc[1], 64))(x)

    # 第一次上采样
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(32, kernel_size=5, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    # 第二次上采样
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(16, kernel_size=5, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    # 第三次上采样
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(1, kernel_size=5, padding='same', activation='linear')(x)  # 修改激活函数为linear

    # 裁剪多余的两个波段以匹配 430
    x = layers.Cropping1D(cropping=(0, 2))(x)  # 从右侧裁剪2个波段

    generator = models.Model(inp, x, name='Generator')
    return generator

def build_discriminator(input_dim):
    inp = layers.Input(shape=(input_dim, 1), name='disc_input')

    x = layers.Conv1D(16, kernel_size=5, strides=2, padding='same')(inp)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(32, kernel_size=5, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(128, kernel_size=5, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    discriminator = models.Model(inp, x, name='Discriminator')
    return discriminator


# ========== 4. 损失函数与优化器 ==========

# 定义损失函数
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def reconstruction_loss(y_true, y_pred):
    """Mean squared error for reconstruction loss."""
    return tf.reduce_mean(tf.square(y_true - y_pred))


def generator_adversarial_loss(discriminator_output):
    """Adversarial loss for generator."""
    return binary_cross_entropy(tf.ones_like(discriminator_output), discriminator_output)


def discriminator_loss(real_output, fake_output):
    """Discriminator loss."""
    real_loss = binary_cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


# ========== 5. 训练步骤定义 ==========

@tf.function
def train_step(generator, discriminator, optimizer_G, optimizer_D, real_data, lambda_rec):
    # 训练判别器
    with tf.GradientTape() as tape_D:
        fake_data = generator(real_data, training=True)
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(fake_data, training=True)
        loss_D = discriminator_loss(real_output, fake_output)
    gradients_D = tape_D.gradient(loss_D, discriminator.trainable_variables)
    optimizer_D.apply_gradients(zip(gradients_D, discriminator.trainable_variables))

    # 训练生成器
    with tf.GradientTape() as tape_G:
        reconstructed = generator(real_data, training=True)
        fake_output = discriminator(reconstructed, training=False)
        loss_reconstruction = reconstruction_loss(real_data, reconstructed)
        loss_adv = generator_adversarial_loss(fake_output)
        loss_G = lambda_rec * loss_reconstruction + loss_adv
    gradients_G = tape_G.gradient(loss_G, generator.trainable_variables)
    optimizer_G.apply_gradients(zip(gradients_G, generator.trainable_variables))

    return loss_D, loss_G


# ========== 6. 主训练循环 ==========

def train_AEAN_Rx(data_3d, epochs=50, batch_size=256, lambda_rec=10.0):
    generator = build_generator(input_dim=data_3d.shape[1])
    discriminator = build_discriminator(input_dim=data_3d.shape[1])

    # 优化器
    optimizer_G = optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    optimizer_D = optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

    # 数据集
    dataset = tf.data.Dataset.from_tensor_slices(data_3d).shuffle(buffer_size=1024).batch(batch_size)

    # 训练日志
    history = {'d_loss': [], 'g_loss': []}

    for epoch in range(1, epochs + 1):
        d_loss_epoch = 0.0
        g_loss_epoch = 0.0
        batch_count = 0

        for batch in dataset:
            d_loss, g_loss = train_step(generator, discriminator, optimizer_G, optimizer_D, batch, lambda_rec)
            d_loss_epoch += d_loss
            g_loss_epoch += g_loss
            batch_count += 1

        avg_d_loss = d_loss_epoch / batch_count
        avg_g_loss = g_loss_epoch / batch_count
        history['d_loss'].append(avg_d_loss.numpy())
        history['g_loss'].append(avg_g_loss.numpy())

        print(f"[Epoch {epoch}/{epochs}] [D loss: {avg_d_loss:.4f}] [G loss: {avg_g_loss:.4f}]")

    return generator, discriminator, history


# ========== 7. 异常检测与 RX 算法优化 ==========

def anomaly_detection(generator, data_3d, H, W, L, mean, std, threshold_percentile=97):
    # 重构测试数据
    reconstructed_data = generator.predict(data_3d, batch_size=256)
    residual = data_3d.reshape(-1, L) - reconstructed_data.reshape(-1, L)

    # 反标准化残差
    residual = residual * std.reshape(1, L)  # 如果需要

    # 计算 RX 残差
    mean_residual = np.mean(residual, axis=0)
    cov_residual = np.cov(residual, rowvar=False) + 1e-6 * np.eye(L)  # 加小值防止奇异
    inv_cov_residual = np.linalg.inv(cov_residual)

    # 马氏距离计算
    diff = residual - mean_residual
    mahal_dist = np.einsum('ij,jk,ik->i', diff, inv_cov_residual, diff)

    # 阈值判定
    threshold = np.percentile(mahal_dist, threshold_percentile)
    anomaly_map = (mahal_dist > threshold).astype(int)
    anomaly_map_2d = anomaly_map.reshape(H, W)

    # 可视化
    plt.figure(figsize=(10, 10))
    plt.imshow(anomaly_map_2d, cmap='gray')
    plt.title('Anomaly Detection Map')
    plt.axis('off')
    plt.show()

    return anomaly_map_2d


# ========== 8. 主程序 ==========

def main():
    # 数据路径
    img_file = '../correction/frt0000c9db_07_if164l_trr3_corr_p.img'  # 替换为您的 .img 文件路径
    hdr_file = '../correction/frt0000c9db_07_if164l_trr3_corr_p.hdr'  # 替换为您的 .hdr 文件路径

    # 加载和预处理数据
    data_3d, H, W, L, mean, std = load_and_preprocess_data(img_file, hdr_file)

    # 设置训练参数
    batch_size = 256
    epochs = 100  # 增加训练轮次以提高模型性能
    lambda_rec = 10.0  # 重构损失的权重

    # 训练 AEAN-RX 模型
    generator, discriminator, history = train_AEAN_Rx(data_3d, epochs, batch_size, lambda_rec)

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curves')
    plt.show()

    # 进行异常检测
    anomaly_map = anomaly_detection(generator, data_3d, H, W, L, mean, std, threshold_percentile=97)

    # 输出结果
    print("Anomaly detection completed.")


if __name__ == "__main__":
    main()

import spectral
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening, binary_closing
import os

# ========== 1. 数据加载与预处理 ==========

def load_and_preprocess_data(img_file, hdr_file):
    crism_data = spectral.open_image(hdr_file).load()

    # MinMax归一化
    min_val = np.min(crism_data, axis=(0, 1), keepdims=True)
    max_val = np.max(crism_data, axis=(0, 1), keepdims=True)
    crism_data = (crism_data - min_val) / (max_val - min_val + 1e-8)

    H, W, L = crism_data.shape
    data_2d = crism_data.reshape(-1, L).astype(np.float32)
    data_3d = data_2d.reshape(-1, L, 1)

    return data_3d, H, W, L, min_val, max_val


# ========== 2. 构建生成器 (Autoencoder) ==========

def build_generator(input_dim):
    inp = layers.Input(shape=(input_dim, 1), name='gen_input')

    # Encoder
    x = layers.Conv1D(32, kernel_size=7, strides=2, padding='same')(inp)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(128, kernel_size=3, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)

    # Bottleneck
    shape_before_fc = tf.keras.backend.int_shape(x)
    x = layers.Flatten()(x)
    x = layers.Dense(shape_before_fc[1] * 128, activation='relu')(x)

    # Decoder
    x = layers.Dense(shape_before_fc[1] * 128, activation='relu')(x)
    x = layers.Reshape((shape_before_fc[1], 128))(x)

    # 第一次上采样：形状恢复到更高维度
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(64, kernel_size=5, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    # 第二次上采样
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(32, kernel_size=5, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    # 第三次上采样
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(16, kernel_size=5, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    # 输出层
    x = layers.Conv1D(1, kernel_size=5, padding='same', activation='linear')(x)

    # 裁剪多余的两个波段以匹配430
    x = layers.Cropping1D(cropping=(0, 2))(x)  # 从右侧裁剪2个波段

    generator = models.Model(inp, x, name='Generator')
    return generator


# ========== 3. 构建判别器 (Discriminator) ==========

def build_discriminator(input_dim):
    inp = layers.Input(shape=(input_dim, 1), name='disc_input')

    x = layers.Conv1D(16, kernel_size=5, strides=2, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inp)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(32, kernel_size=5, strides=2, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(128, kernel_size=5, strides=2, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    discriminator = models.Model(inp, x, name='Discriminator')
    return discriminator


# ========== 4. 损失函数与优化器 ==========

# 定义损失函数
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def reconstruction_loss(y_true, y_pred, band_weights=None):
    """加权均方误差重构损失."""
    if band_weights is not None:
        return tf.reduce_mean(tf.square(y_true - y_pred) * band_weights)
    else:
        return tf.reduce_mean(tf.square(y_true - y_pred))


def generator_adversarial_loss(discriminator_output):
    """生成器的对抗损失."""
    return binary_cross_entropy(tf.ones_like(discriminator_output), discriminator_output)


def discriminator_loss(real_output, fake_output):
    """判别器的损失."""
    real_loss = binary_cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


# ========== 5. 训练步骤定义 ==========

@tf.function
def train_step(generator, discriminator, optimizer_G, optimizer_D, real_data, lambda_rec, band_weights=None,
               noise_factor=0.01):
    # 添加噪声
    noisy_real_data = real_data + noise_factor * tf.random.normal(shape=tf.shape(real_data))

    # 训练判别器
    with tf.GradientTape() as tape_D:
        fake_data = generator(noisy_real_data, training=True)
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(fake_data, training=True)
        loss_D = discriminator_loss(real_output, fake_output)
    gradients_D = tape_D.gradient(loss_D, discriminator.trainable_variables)
    optimizer_D.apply_gradients(zip(gradients_D, discriminator.trainable_variables))

    # 训练生成器
    with tf.GradientTape() as tape_G:
        reconstructed = generator(noisy_real_data, training=True)
        fake_output = discriminator(reconstructed, training=False)
        loss_reconstruction = reconstruction_loss(real_data, reconstructed, band_weights=band_weights)
        loss_adv = generator_adversarial_loss(fake_output)
        loss_G = lambda_rec * loss_reconstruction + loss_adv
    gradients_G = tape_G.gradient(loss_G, generator.trainable_variables)
    optimizer_G.apply_gradients(zip(gradients_G, generator.trainable_variables))

    return loss_D, loss_G


# ========== 6. 主训练循环 ==========

def train_AEAN_Rx(data_3d, epochs=60, batch_size=512, lambda_rec=100.0):
    generator = build_generator(input_dim=data_3d.shape[1])
    discriminator = build_discriminator(input_dim=data_3d.shape[1])

    # 使用更低的学习率
    optimizer_G = optimizers.Adam(learning_rate=5e-5, beta_1=0.5)
    optimizer_D = optimizers.Adam(learning_rate=5e-5, beta_1=0.5)

    # 数据集
    dataset = tf.data.Dataset.from_tensor_slices(data_3d).shuffle(buffer_size=1024).batch(batch_size)

    # 训练日志
    history = {'d_loss': [], 'g_loss': []}

    # 回调函数模拟（保留早停）
    early_stopping = EarlyStopping(monitor='g_loss', patience=10, restore_best_weights=True)

    # --------------------------------------------
    # 这里移除 ReduceLROnPlateau，改为手动衰减逻辑
    # --------------------------------------------

    for epoch in range(1, epochs + 1):
        d_loss_epoch = 0.0
        g_loss_epoch = 0.0
        batch_count = 0

        for batch in dataset:
            d_loss, g_loss = train_step(
                generator, discriminator,
                optimizer_G, optimizer_D,
                batch,
                lambda_rec
            )
            d_loss_epoch += d_loss
            g_loss_epoch += g_loss
            batch_count += 1

        # 计算该 epoch 的平均损失
        avg_d_loss = d_loss_epoch / batch_count
        avg_g_loss = g_loss_epoch / batch_count

        # 记录历史
        history['d_loss'].append(avg_d_loss.numpy())
        history['g_loss'].append(avg_g_loss.numpy())

        print(f"[Epoch {epoch}/{epochs}] "
              f"[D loss: {avg_d_loss:.4f}] "
              f"[G loss: {avg_g_loss:.4f}]")

        # -----------------------------
        # # 1) 早停判断
        # # -----------------------------
        # # 如果 epoch > 10 且 G loss 创最近10个 epoch 的最低值 → 更新 early_stopping 计数器
        # if epoch > 10 and avg_g_loss < np.min(history['g_loss'][-10:]):
        #     early_stopping.on_epoch_end(epoch, logs={'g_loss': avg_g_loss})
        #
        # # -----------------------------
        # # 2) 学习率衰减逻辑（手动替代 ReduceLROnPlateau）
        # # -----------------------------
        # # 如果 epoch > 5 且本 epoch 的 g_loss 比最近 5 个 epoch 的均值还要高，
        # # 则视为模型可能过拟合或发散，衰减学习率
        # if epoch > 5 and avg_g_loss > np.mean(history['g_loss'][-5:]):
        #     factor = 0.5  # 衰减因子
        #     min_lr = 1e-6  # 学习率下限
        #
        #     current_lr_G = optimizer_G.learning_rate.numpy()
        #     new_lr_G = max(current_lr_G * factor, min_lr)
        #     optimizer_G.learning_rate.assign(new_lr_G)
        #
        #     current_lr_D = optimizer_D.learning_rate.numpy()
        #     new_lr_D = max(current_lr_D * factor, min_lr)
        #     optimizer_D.learning_rate.assign(new_lr_D)
        #
        #     print(f"  [LR Decay] G lr: {current_lr_G:.6f} -> {new_lr_G:.6f}, "
        #           f"D lr: {current_lr_D:.6f} -> {new_lr_D:.6f}")
        #
        # # -----------------------------
        # # 3) 检查早停
        # # -----------------------------
        # if early_stopping.stopped_epoch > 0:
        #     print("Early stopping triggered.")
        #     break
        if epoch % 10 == 0:
            factor = 0.5  # 衰减因子
            min_lr = 1e-6  # 学习率下限

            # 更新生成器的学习率
            current_lr_G = optimizer_G.learning_rate.numpy()
            new_lr_G = max(current_lr_G * factor, min_lr)
            optimizer_G.learning_rate.assign(new_lr_G)

            # 更新判别器的学习率
            current_lr_D = optimizer_D.learning_rate.numpy()
            new_lr_D = max(current_lr_D * factor, min_lr)
            optimizer_D.learning_rate.assign(new_lr_D)

            print(f"  [LR Decay] Epoch {epoch}: G lr decayed from {current_lr_G:.6f} to {new_lr_G:.6f}, "
                  f"D lr decayed from {current_lr_D:.6f} to {new_lr_D:.6f}")

    return generator, discriminator, history


# ========== 7. 异常检测与 RX 算法优化 ==========

def anomaly_detection(generator, data_3d, H, W, L, mean, std, threshold_percentile=96):
    # 重构测试数据
    reconstructed_data = generator.predict(data_3d, batch_size=256)
    residual = data_3d.reshape(-1, L) - reconstructed_data.reshape(-1, L)

    # 反标准化残差（如果需要）
    residual = residual * std.reshape(1, L)

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

    # 加强协方差正则化
    cov_residual = np.cov(residual, rowvar=False) + 1e-4 * np.eye(L)  # 增大正则化系数

    # 可视化
    plt.figure(figsize=(10, 10))
    plt.imshow(anomaly_map_2d, cmap='gray')
    plt.title(f'Anomaly Detection Map (Threshold: {threshold_percentile}%)')
    plt.axis('off')
    plt.show()

    return anomaly_map_2d





# ==========8. 主程序 ==========

def main():
        # 数据路径
        img_file = '../reproject/frt0000c9db_07_if164l_trr3_corr.img'  # 替换为您的 .img 文件路径
        hdr_file = '../reproject/frt0000c9db_07_if164l_trr3_corr.hdr'  # 替换为您的 .hdr 文件路径

        # 创建保存目录
        save_dir = './results'
        os.makedirs(save_dir, exist_ok=True)

        # 加载和预处理数据
        data_3d, H, W, L, mean, std = load_and_preprocess_data(img_file, hdr_file)

        # 设置训练参数
        batch_size = 256
        epochs = 100
        lambda_rec = 30.0

        # 训练 AEAN-RX 模型
        generator, discriminator, history = train_AEAN_Rx(data_3d, epochs, batch_size, lambda_rec)

        # 保存生成器和判别器模型
        generator.save(os.path.join(save_dir, 'generator_model.h5'))
        discriminator.save(os.path.join(save_dir, 'discriminator_model.h5'))
        print(f"Models saved to {save_dir}")

        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(history['d_loss'], label='Discriminator Loss')
        plt.plot(history['g_loss'], label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss Curves')
        plt.savefig(os.path.join(save_dir, 'training_loss_curves.png'))
        plt.show()

        # 进行异常检测
        anomaly_map = anomaly_detection(generator, data_3d, H, W, L, mean, std, threshold_percentile=92)

        # 保存原始异常检测结果
        np.save(os.path.join(save_dir, 'anomaly_map_raw_3.npy'), anomaly_map)
        plt.imsave(os.path.join(save_dir, 'anomaly_map_raw_3.png'), anomaly_map, cmap='gray')
        print(f"Raw anomaly map saved to {save_dir}")

    
        # 输出结果
        print("Anomaly detection completed.")

if __name__ == "__main__":
        main()
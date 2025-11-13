import tensorflow as tf

"""
ConvBlock: a robust modular convolutional building unit.
- Combines convolution, normalization, activation, and pooling.

Padding='same' preserves spatial dimensions, aiding skip connections and stability.

Normalization layer improves convergence and stabilizes training dynamics.
    BatchNormalization: Ioffe & Szegedy, "Batch Normalization" (ICML 2015).
        - normalizes across batch+spatial dims — helps accelerate training.
    LayerNormalization: Ba et al., "Layer Normalization" (arXiv:1607.06450).
        - normalizes per sample — stable for small batch sizes or transformers.
    GroupNormalization: Wu & He, "Group Normalization" (ECCV 2018).
        - balances BatchNorm’s stability and LayerNorm’s independence.
        - robust to varying batch sizes and enhances feature consistency.
        - learnable scale (gamma) and shift (beta) allow rescaling after normalization.
    InstanceNormalization: Ulyanov et al., "Instance Normalization" (arXiv:1607.08022).
        - strong in style transfer and low-level vision.
        - strongly stabilizes low-level texture and color statistics.
        - normalizes per-channel per-image, improving invariance to contrast and lighting.
        - commonly used in style transfer and anomaly detection.

Activation: introduces non-linearity for expressive representations.
    ReLU: prevents vanishing gradients while keeping computational cost low.
        Glorot et al., “Deep Sparse Rectifier Neural Networks” (AISTATS 2011)
    LeakyReLU: Allows small negative gradients to flow (e.g., 0.01x), reducing “dead neuron” problem of ReLU.
        Maas et al., “Rectifier Nonlinearities Improve Neural Network Acoustic Models” (ICML 2013)
    ELU: Smoothly saturates for negative values, improving robustness and faster learning with zero-mean activations.
        Clevert et al., “Fast and Accurate Deep Network Learning by Exponential Linear Units” (ICLR 2016)
    SELU: Self-normalizing: automatically keeps activations with zero mean and unit variance — improves training stability in deep nets.
        Klambauer et al., “Self-Normalizing Neural Networks” (NIPS 2017)
    GELU: Weights inputs by their probability of being positive, improving smoothness and uncertainty modeling. Used in BERT and Vision Transformers.
        Hendrycks & Gimpel, “Gaussian Error Linear Units (GELUs)” (arXiv:1606.08415)
    Tanh: Scales activations between -1 and 1, centering data and helping optimization in smaller or shallow networks.
        LeCun et al., “Efficient BackProp” (1998)
    Sigmoid: Maps values to [0, 1]; useful for probabilistic outputs but prone to vanishing gradients.
        Cybenko, “Approximation by Superpositions of Sigmoidal Functions” (1989)
    Softplus: Smooth approximation to ReLU, providing continuous gradients and numerical stability in some cases.
        Dugas et al., “Incorporating Second-Order Functional Knowledge for Better Option Pricing” (NIPS 2001)
    Hard Sigmoid/Hard Swish: Piecewise linear approximations to sigmoid/swish; cheaper and often used in mobile/efficient networks.
        Howard et al., “Searching for MobileNetV3” (ICCV 2019)

Pooling: provides translational invariance and reduces overfitting.
    Max pooling captures salient features
    Average pooling smooths activation noise.
    Combined pooling enhances both detail retention and stability.
        - Reduces overfitting to specific local extrema (max) while preserving smoothness (avg)
        - Provides better gradient flow and feature diversity.
        - Elementwise mean improves feature consistency and stability across augmentations.

Dropout: regularizes and improves robustness against overfitting.

ResidualBlock: mitigates vanishing gradients and improves convergence.
    He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016).

    Blocks: Stacks ConvBlocks for deeper hierarchical feature extraction.
    Shortcut: 1x1 conv aligns dimensions for residual addition.
    Activation: Element-wise addition helps gradient flow and feature reuse.

DenseBlock: compact fully connected feature transformation.
    Srivastava et al., "Dropout: A Simple Way to Prevent Overfitting" (JMLR 2014).


SEBlock: Squeeze-and-Excitation Block (Hu et al., CVPR 2018)
    Improves channel-wise feature recalibration.
    Improves channel selectivity and representation quality, especially for defect localization.
    Increases gradient stability by amplifying relevant feature channels.

CBAM: (Woo et al., ECCV 2018)
    Sequential Channel + Spatial attention for feature refinement.
    Reduces false positives in defect detection by highlighting relevant spatial regions.
    Improves both channel- and spatial-wise feature selectivity.

SelfAttention:
    Non-local Self-Attention (Wang et al., CVPR 2018)
    Captures long-range spatial dependencies in feature maps.
    Increases consistency on highly variable patterns (e.g., subtle defect textures).
    Improves feature aggregation and gradient flow stability by bypassing local bottlenecks.

SpatialAttentionLite: (Lightweight Spatial Attention (Efficient Variant))
    Lightweight spatial attention using 3x3 depthwise convolutions.
    Adds spatial focus with minimal cost (for real-time segmentation).


"""


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, activation='relu',
                 normalization='batch', pooling=None, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same')

        self.norm = None
        if normalization == 'batch':
            self.norm = tf.keras.layers.BatchNormalization()
        elif normalization == 'layer':
            self.norm = tf.keras.layers.LayerNormalization()
        elif normalization == 'group': # Implement Group Normalization
            self.norm = GroupNormalization(groups=groups)
        elif normalization == 'instance': # Instance normalization = GroupNorm with groups = num_channels
            self.norm = InstanceNormalization()
        elif normalization is not None:
            raise ValueError(f'Unknown normalization type: {normalization}')

        self.activation = tf.keras.layers.Activation(activation)

        self.pool = None
        if pooling == 'max':
            self.pool = tf.keras.layers.MaxPool2D(pool_size=2)
        elif pooling == 'avg':
            self.pool = tf.keras.layers.AvgPool2D(pool_size=2)
        elif pooling == 'both':
            self.pool = CombinedPooling(pool_size=2)
        elif pooling is not None:
            raise ValueError(f'Unknown pooling type: {pooling}')

    def call(self, x, training=False):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x, training=training)
        x = self.activation(x)
        if self.pool:
            x = self.pool(x)
        return x


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, blocks=2, activation='relu', normalization='batch', **kwargs):
        super().__init__(**kwargs)
        self.blocks = []
        for _ in range(blocks):
            self.blocks.append(ConvBlock(filters, activation=activation, normalization=normalization))
        self.shortcut = tf.keras.layers.Conv2D(filters, 1, padding='same')

    def call(self, x, training=False):
        shortcut = self.shortcut(x)
        for block in self.blocks:
            x = block(x, training=training)
        return tf.keras.activations.relu(x + shortcut)


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, units, activation='relu', dropout=None, **kwargs):
        super().__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(units)
        self.activation = tf.keras.layers.Activation(activation)
        self.dropout = tf.keras.layers.Dropout(dropout) if dropout else None

    def call(self, x, training=False):
        x = self.fc(x)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x, training=training)
        return x

# segmentation blocks
class UpConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=2, activation='relu'):
        super().__init__()
        self.up = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same')
        self.activation = tf.keras.layers.Activation(activation)

    def call(self, x, training=False):
        return self.activation(self.up(x))

# attention blocks
class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.g = tf.keras.layers.Conv2D(filters, 1, padding='same')
        self.x = tf.keras.layers.Conv2D(filters, 1, padding='same')
        self.psi = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')

    def call(self, g_input, x_input):
        g1 = self.g(g_input)
        x1 = self.x(x_input)
        psi = tf.keras.activations.sigmoid(g1 + x1)
        return x_input * psi

class SEBlock(tf.keras.layers.Layer):
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(input_shape[-1] // self.ratio, activation='relu')
        self.fc2 = tf.keras.layers.Dense(input_shape[-1], activation='sigmoid')

    def call(self, x):
        se = self.global_pool(x)
        se = self.fc1(se)
        se = self.fc2(se)
        se = tf.reshape(se, [-1, 1, 1, tf.shape(x)[-1]])
        return x * se

class CBAM(tf.keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.channel_avg = tf.keras.layers.GlobalAveragePooling2D()
        self.channel_max = tf.keras.layers.GlobalMaxPooling2D()
        self.fc1 = tf.keras.layers.Dense(input_shape[-1] // self.ratio, activation='relu')
        self.fc2 = tf.keras.layers.Dense(input_shape[-1])

        self.spatial_conv = tf.keras.layers.Conv2D(1, 7, padding='same', activation='sigmoid')

    def call(self, x):
        # Channel attention
        avg = self.fc2(self.fc1(self.channel_avg(x)))
        max_ = self.fc2(self.fc1(self.channel_max(x)))
        chn_att = tf.nn.sigmoid(avg + max_)
        chn_att = tf.reshape(chn_att, [-1, 1, 1, tf.shape(x)[-1]])
        x = x * chn_att

        # Spatial attention
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        spat_att = self.spatial_conv(concat)
        return x * spat_att

class SelfAttention2D(tf.keras.layers.Layer):
    def __init__(self, filters=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        c = input_shape[-1]
        f = self.filters or c // 8
        self.theta = tf.keras.layers.Conv2D(f, 1, padding='same')
        self.phi = tf.keras.layers.Conv2D(f, 1, padding='same')
        self.g = tf.keras.layers.Conv2D(f, 1, padding='same')
        self.out_conv = tf.keras.layers.Conv2D(c, 1, padding='same')

    def call(self, x):
        batch, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        theta = tf.reshape(self.theta(x), [batch, -1, c // 8])
        phi = tf.reshape(self.phi(x), [batch, -1, c // 8])
        attn = tf.nn.softmax(tf.matmul(theta, phi, transpose_b=True))
        g = tf.reshape(self.g(x), [batch, -1, c // 8])
        attn_out = tf.matmul(attn, g)
        attn_out = tf.reshape(attn_out, [batch, h, w, c // 8])
        out = self.out_conv(attn_out)
        return x + out

class SpatialAttentionLite(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.depthwise = tf.keras.layers.DepthwiseConv2D(3, padding='same', activation='sigmoid')

    def call(self, x):
        return x * self.depthwise(x)

# helper classes
class GroupNormalization(tf.keras.layers.Layer):
    """Implements Group Normalization (https://arxiv.org/abs/1803.08494)"""
    def __init__(self, groups=8, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.epsilon = epsilon

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels % self.groups != 0:
            raise ValueError(f'Number of channels ({channels}) must be divisible by groups ({self.groups}).')
        self.gamma = self.add_weight(shape=(channels,), initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(channels,), initializer='zeros', trainable=True)

    def call(self, x, training=False):
        input_shape = tf.shape(x)
        N, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        G = self.groups
        x = tf.reshape(x, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.epsilon)
        x = tf.reshape(x, [N, H, W, C])
        return self.gamma * x + self.beta


class InstanceNormalization(GroupNormalization):
    """Instance normalization = GroupNorm with G = C"""
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(groups=None, epsilon=epsilon, **kwargs)

    def build(self, input_shape):
        self.groups = input_shape[-1]
        super().build(input_shape)


class CombinedPooling(tf.keras.layers.Layer):
    """Combines Max and Average pooling (elementwise mean)."""
    def __init__(self, pool_size=2, **kwargs):
        super().__init__(**kwargs)
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=pool_size)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=pool_size)

    def call(self, x):
        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)
        return 0.5 * (max_pooled + avg_pooled)
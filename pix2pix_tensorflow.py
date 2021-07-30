#%%
import tensorflow as tf

import os
import time

from matplotlib import pyplot as plt
from IPython import display

import os
import librosa
import math
import numpy as np
import pydot

from skimage.io import imshow
#%% spyder용 GPU 유무 확인 코드, Colab은 따로 적용 X
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
#%%
clean_path = 'C:/Users/NGV32/.spyder-py3/hyundai_project/data/clean_wav/'
noise_path = 'C:/Users/NGV32/.spyder-py3/hyundai_project/data/mix_wav/'

clean_list = os.listdir(clean_path)
noise_list = os.listdir(noise_path)

#%%
def normalization(data, a, b):
    normal = (b - a)*((data - np.min(data)) / (np.max(data) - np.min(data))) + a
    return normal

def denormalization(data, a, b, Max, Min): # GAN 값 도출 이후 역정규화를 통해 STFT로 변환하기 위한 함수
    denorm = ((data - a) * (Max - Min))/(b - a) + Min
    return denorm

#%%
hop_length = 257  # number of samples per time-step in spectrogram
n_fft = 510

noise = np.zeros((len(noise_list), 256, 256, 2)).astype('float32')
clean = np.zeros((len(clean_list), 256, 256, 2)).astype('float32')
#%%
AB_length = 0
AX_length = 0
BA_length = 0
BD_length = 0
DN_length = 0
KS_length = 0
SU_length = 0
KP_length = 0

for i in range(len(noise_list)):
    if noise_list[i][0:2] == 'AB':
        AB_length += 1
    if noise_list[i][0:2] == 'AX':
        AX_length += 1
    if noise_list[i][0:2] == 'BA':
        BA_length += 1
    if noise_list[i][0:2] == 'BD':
        BD_length += 1
    if noise_list[i][0:2] == 'DN':
        DN_length += 1
    if noise_list[i][0:2] == 'KS':
        KS_length += 1
    if noise_list[i][0:2] == 'SU':
        SU_length += 1
    if noise_list[i][0:2] == '카펙':
        KP_length += 1
#%%
AB_noise = np.zeros((AB_length, 256, 256, 2)).astype('float32')
AX_noise = np.zeros((AX_length, 256, 256, 2)).astype('float32')
BA_noise = np.zeros((BA_length, 256, 256, 2)).astype('float32')
BD_noise = np.zeros((BD_length, 256, 256, 2)).astype('float32')
DN_noise = np.zeros((DN_length, 256, 256, 2)).astype('float32')
KS_noise = np.zeros((KS_length, 256, 256, 2)).astype('float32')
SU_noise = np.zeros((SU_length, 256, 256, 2)).astype('float32')
KP_noise = np.zeros((KP_length, 256, 256, 2)).astype('float32')

AB_clean = np.zeros((AB_length, 256, 256, 2)).astype('float32')
AX_clean = np.zeros((AX_length, 256, 256, 2)).astype('float32')
BA_clean = np.zeros((BA_length, 256, 256, 2)).astype('float32')
BD_clean = np.zeros((BD_length, 256, 256, 2)).astype('float32')
DN_clean = np.zeros((DN_length, 256, 256, 2)).astype('float32')
KS_clean = np.zeros((KS_length, 256, 256, 2)).astype('float32')
SU_clean = np.zeros((SU_length, 256, 256, 2)).astype('float32')
KP_clean = np.zeros((KP_length, 256, 256, 2)).astype('float32')
#%%
noi_mag_max = []
noi_mag_min = []
noi_ang_max = []
noi_ang_min = []

cle_mag_max = []
cle_mag_min = []
cle_ang_max = []
cle_ang_min = []

AB, AX, BA, BD, DN, KS, SU, KP = 0, 0, 0, 0, 0, 0, 0, 0

for i in range(len(noise_list)):
    
    noi, sr = librosa.load(noise_path + noise_list[i], sr=16384)
    cle, sr = librosa.load(clean_path + clean_list[i], sr=16384)

    stft_noise = librosa.stft(noi, n_fft=n_fft, hop_length=hop_length)
    stft_clean = librosa.stft(cle, n_fft=n_fft, hop_length=hop_length)
    
    magnitude_noise = np.abs(stft_noise)
    magnitude_clean = np.abs(stft_clean)
    
    log_magnitude_noise = librosa.amplitude_to_db(magnitude_noise)
    log_magnitude_clean = librosa.amplitude_to_db(magnitude_clean)

    noi_mag_max.append(np.max(log_magnitude_noise))
    noi_mag_min.append(np.max(log_magnitude_noise))

    cle_mag_max.append(np.max(log_magnitude_clean))
    cle_mag_min.append(np.min(log_magnitude_clean))

    angle_noise = np.unwrap(np.angle(stft_noise))
    angle_clean = np.unwrap(np.angle(stft_clean))

    noi_ang_max.append(np.max(angle_noise))
    noi_ang_min.append(np.min(angle_noise))

    cle_ang_max.append(np.max(angle_clean))
    cle_ang_min.append(np.max(angle_clean))

    log_magnitude_noise = normalization(log_magnitude_noise, -1, 1)
    log_magnitude_clean = normalization(log_magnitude_clean, -1, 1)
    angle_noise = normalization(angle_noise, -1, 1)
    angle_clean = normalization(angle_clean, -1, 1)
    
    if noise_list[i][0:2] == 'AB':
        AB_noise[AB,:,:,0] = log_magnitude_noise.astype('float32')
        AB_clean[AB,:,:,0] = log_magnitude_clean.astype('float32')
        
        AB_noise[AB,:,:,1] = angle_noise.astype('float32')
        AB_clean[AB,:,:,1] = angle_clean.astype('float32')
        AB += 1
        
    if noise_list[i][0:2] == 'AX':
        AX_noise[AX,:,:,0] = log_magnitude_noise.astype('float32')
        AX_clean[AX,:,:,0] = log_magnitude_clean.astype('float32')
        
        AX_noise[AX,:,:,1] = angle_noise.astype('float32')
        AX_clean[AX,:,:,1] = angle_clean.astype('float32')
        AX += 1
        
    if noise_list[i][0:2] == 'BA':
        BA_noise[BA,:,:,0] = log_magnitude_noise.astype('float32')
        BA_clean[BA,:,:,0] = log_magnitude_clean.astype('float32')
        
        BA_noise[BA,:,:,1] = angle_noise.astype('float32')
        BA_clean[BA,:,:,1] = angle_clean.astype('float32')
        BA += 1
        
    if noise_list[i][0:2] == 'BD':
        BD_noise[BD,:,:,0] = log_magnitude_noise.astype('float32')
        BD_clean[BD,:,:,0] = log_magnitude_clean.astype('float32')
        
        BD_noise[BD,:,:,1] = angle_noise.astype('float32')
        BD_clean[BD,:,:,1] = angle_clean.astype('float32')
        BD += 1
        
    if noise_list[i][0:2] == 'DN':
        DN_noise[DN,:,:,0] = log_magnitude_noise.astype('float32')
        DN_clean[DN,:,:,0] = log_magnitude_clean.astype('float32')
        
        DN_noise[DN,:,:,1] = angle_noise.astype('float32')
        DN_clean[DN,:,:,1] = angle_clean.astype('float32')
        DN += 1
        
    if noise_list[i][0:2] == 'KS':
        KS_noise[KS,:,:,0] = log_magnitude_noise.astype('float32')
        KS_clean[KS,:,:,0] = log_magnitude_clean.astype('float32')
        
        KS_noise[KS,:,:,1] = angle_noise.astype('float32')
        KS_clean[KS,:,:,1] = angle_clean.astype('float32')
        KS += 1
        
    if noise_list[i][0:2] == 'SU':
        SU_noise[SU,:,:,0] = log_magnitude_noise.astype('float32')
        SU_clean[SU,:,:,0] = log_magnitude_clean.astype('float32')
        
        SU_noise[SU,:,:,1] = angle_noise.astype('float32')
        SU_clean[SU,:,:,1] = angle_clean.astype('float32')
        SU += 1
        
    if noise_list[i][0:2] == '카펙':
        KP_noise[KP,:,:,0] = log_magnitude_noise.astype('float32')
        KP_clean[KP,:,:,0] = log_magnitude_clean.astype('float32')
        
        KP_noise[KP,:,:,1] = angle_noise.astype('float32')
        KP_clean[KP,:,:,1] = angle_clean.astype('float32')
        KP += 1
        
'''  
    noise[i,:,:,0] = log_magnitude_noise.astype('float32')
    clean[i,:,:,0] = log_magnitude_clean.astype('float32')
    
    noise[i,:,:,1] = angle_noise.astype('float32')
    clean[i,:,:,1] = angle_clean.astype('float32')
'''
#%%
'''
def load_image_train(inp, real):
    
  input_image = normalization(inp, -1, 1).astype('float32')
  real_image = normalization(real, -1, 1).astype('float32')

  return input_image, real_image
'''
#%%
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
#%%
noise_ab_train = AB_noise[:AB_length - 10,:,:,:]
noise_ax_train = AX_noise[:AX_length - 10,:,:,:]
noise_ba_train = BA_noise[:BA_length - 10,:,:,:]
noise_bd_train = BD_noise[:BD_length - 10,:,:,:]
noise_dn_train = DN_noise[:DN_length - 10,:,:,:]
noise_ks_train = KS_noise[:KS_length - 10,:,:,:]
noise_su_train = SU_noise[:SU_length - 10,:,:,:]
noise_kp_train = KP_noise[:KP_length - 10,:,:,:]

noise_ab_test = AB_noise[AB_length - 10:AB_length,:,:,:]
noise_ax_test = AX_noise[AX_length - 10:AX_length,:,:,:]
noise_ba_test = BA_noise[BA_length - 10:BA_length,:,:,:]
noise_bd_test = BD_noise[BD_length - 10:BD_length,:,:,:]
noise_dn_test = DN_noise[DN_length - 10:DN_length,:,:,:]
noise_ks_test = KS_noise[KS_length - 10:KS_length,:,:,:]
noise_su_test = SU_noise[SU_length - 10:SU_length,:,:,:]
noise_kp_test = KP_noise[KP_length - 10:KP_length,:,:,:]

clean_ab_train = AB_clean[:AB_length - 10,:,:,:]
clean_ax_train = AX_clean[:AX_length - 10,:,:,:]
clean_ba_train = BA_clean[:BA_length - 10,:,:,:]
clean_bd_train = BD_clean[:BD_length - 10,:,:,:]
clean_dn_train = DN_clean[:DN_length - 10,:,:,:]
clean_ks_train = KS_clean[:KS_length - 10,:,:,:]
clean_su_train = SU_clean[:SU_length - 10,:,:,:]
clean_kp_train = KP_clean[:KP_length - 10,:,:,:]

clean_ab_test = AB_clean[AB_length - 10:AB_length,:,:,:]
clean_ax_test = AX_clean[AX_length - 10:AX_length,:,:,:]
clean_ba_test = BA_clean[BA_length - 10:BA_length,:,:,:]
clean_bd_test = BD_clean[BD_length - 10:BD_length,:,:,:]
clean_dn_test = DN_clean[DN_length - 10:DN_length,:,:,:]
clean_ks_test = KS_clean[KS_length - 10:KS_length,:,:,:]
clean_su_test = SU_clean[SU_length - 10:SU_length,:,:,:]
clean_kp_test = KP_clean[KP_length - 10:KP_length,:,:,:]
#%%
noise_train = np.concatenate([noise_ab_train, noise_ax_train, noise_ba_train, noise_bd_train, 
                              noise_dn_train,noise_ks_train,noise_su_train,noise_kp_train], 
                             axis = 0)

clean_train = np.concatenate([clean_ab_train, clean_ax_train, clean_ba_train, clean_bd_train, 
                              clean_dn_train, clean_ks_train, clean_su_train, clean_kp_train], 
                             axis = 0)

noise_test = np.concatenate([noise_ab_test, noise_ax_test, noise_ba_test, noise_bd_test, 
                              noise_dn_test, noise_ks_test, noise_su_test, noise_kp_test], 
                             axis = 0)

clean_test = np.concatenate([clean_ab_test, clean_ax_test, clean_ba_test, clean_bd_test, 
                              clean_dn_test, clean_ks_test, clean_su_test, clean_kp_test], 
                             axis = 0)
#%%
train_dataset = tf.data.Dataset.from_tensor_slices((noise_train,clean_train))
print(train_dataset)
#train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

#%%
test_dataset = tf.data.Dataset.from_tensor_slices((noise_test,clean_test))
print(test_dataset)
# test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
#%%
OUTPUT_CHANNELS = 2

#%%
def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', # stride = 2로 인해서 이미지의 사이즈가 반으로 줄어든다.
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

#%%
def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

#%%
def Generator():
  inputs = tf.keras.layers.Input(shape=[256,256,2])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 2)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

#%%
generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

#%%
inp, _ = noise_train, clean_train
gen_output = generator(inp[0][tf.newaxis,...], training=False)
plt.imshow(gen_output[0,:,:,1])
#%%
LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#%%
def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss

#%%
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 2], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 2], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

#%%
discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

#%%
def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

#%%
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

#%%
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

#%%
def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15,15))

  mag_list = [test_input[0][:,:,0], tar[0][:,:,0], prediction[0][:,:,0]]
  title = ['Input Magnitude', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(mag_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

  plt.figure(figsize=(15,15))

  ang_list = [test_input[0][:,:,1], tar[0][:,:,1], prediction[0][:,:,1]]
  title = ['Input Angle', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(ang_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

#%%
for example_input, example_target in test_dataset.take(40):
  generate_images(generator, example_input, example_target)

#%%
EPOCHS = 20
#%%
import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

#%%
@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)

#%%
def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    for example_input, example_target in test_ds.take(1):
      generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)

#%%
fit(train_dataset, EPOCHS, test_dataset)
#%%
# Run the trained model on a few examples from the test dataset
for inp, tar in test_dataset.take(40):
  generate_images(generator, inp, tar)
#%%
import soundfile

count = 0

max_clean = cle_mag_max[196:211] + cle_mag_max[387:402] + cle_mag_max[514:524]
min_clean = cle_mag_min[196:211] + cle_mag_min[387:402] + cle_mag_min[514:524]

for inp, tar in test_dataset.take(40):
  prediction = generator(inp, training = True)
  Max = max_clean[count]
  Min = min_clean[count]

  gen_mag = prediction[0][:,:,0]
  gt_mag = tar[0][:,:,0]

  gen_denorm = denormalization(gen_mag, -1, 1, Max, Min)
  gt_denorm = denormalization(gt_mag, -1, 1, Max, Min)

  gen_db_amp = librosa.db_to_amplitude(gen_denorm)
  gt_db_amp = librosa.db_to_amplitude(gt_denorm)

  gen_wav = librosa.istft(gen_db_amp, hop_length=hop_length, win_length = n_fft)
  gt_wav = librosa.istft(gt_db_amp, hop_length=hop_length, win_length = n_fft)

  soundfile.write('C:/Users/NGV32/.spyder-py3/hyundai_project/test_pix2pix/prediction/gen_wav_{}.wav' .format(count), gen_wav, sr)
  soundfile.write('C:/Users/NGV32/.spyder-py3/hyundai_project/test_pix2pix/groud_truth/gt_wav_{}.wav' .format(count), gt_wav, sr)

  count += 1
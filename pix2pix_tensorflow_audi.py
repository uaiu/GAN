# -*- coding: utf-8 -*-
"""pix2pix_tensorflow.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JO2Ygr4Ed9RqBSxpOhyYzEgpmMcXXF52
"""

# 드라이브에 접근할 수 있도록 아래 코드 입력
from google.colab import drive
drive.mount('/content/drive')

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

#%%
clean_path = '/content/drive/MyDrive/clean_wav/'
noise_path = '/content/drive/MyDrive/mix_wav/'

clean_list = os.listdir(clean_path)
noise_list = os.listdir(noise_path)

clean_list.sort()
noise_list.sort()

print(noise_list)
print(clean_list)

#%%
def normalization(noise, clean, a, b, Max, Min):
    clean_normal = ((b - a)*(clean - Min)) / (Max - Min) + a
    noise_normal = ((b - a)*(noise - Min)) / (Max - Min) + a
    return noise_normal, clean_normal

def denormalization(data, a, b, Max, Min): # GAN 값 도출 이후 역정규화를 통해 STFT로 변환하기 위한 함수
    denorm = ((data - a) * (Max - Min))/(b - a) + Min
    return denorm

#%%
hop_length = 257  # number of samples per time-step in spectrogram
n_fft = 510

noise = np.zeros((len(noise_list), 256, 256, 2)).astype('float32')
clean = np.zeros((len(clean_list), 256, 256, 2)).astype('float32')

mag_max = []
mag_min = []
ang_max = []
ang_min = []

for i in range(len(noise)):
    noi, sr = librosa.load(noise_path + noise_list[i], sr=16384)
    cle, sr = librosa.load(clean_path + clean_list[i], sr=16384)

    stft_noise = librosa.stft(noi, n_fft=n_fft, hop_length=hop_length)
    stft_clean = librosa.stft(cle, n_fft=n_fft, hop_length=hop_length)
    
    magnitude_noise = np.abs(stft_noise)
    magnitude_clean = np.abs(stft_clean)
    
    log_magnitude_noise = librosa.amplitude_to_db(magnitude_noise)
    log_magnitude_clean = librosa.amplitude_to_db(magnitude_clean)

    angle_noise = np.unwrap(np.angle(stft_noise))
    angle_clean = np.unwrap(np.angle(stft_clean))

    mag_max.append(np.max((log_magnitude_noise, log_magnitude_clean)))
    mag_min.append(np.min((log_magnitude_noise, log_magnitude_clean)))

    ang_max.append(np.max((angle_noise, angle_clean)))
    ang_min.append(np.min((angle_noise, angle_clean)))

    log_magnitude_noise, log_magnitude_clean = normalization(log_magnitude_noise, log_magnitude_clean, -1, 1, np.max((log_magnitude_noise, log_magnitude_clean)), np.min((log_magnitude_noise, log_magnitude_clean)))

    angle_noise, angle_clean = normalization(angle_noise, angle_clean, -1, 1, np.max((angle_noise, angle_clean)), np.min((angle_noise, angle_clean)))
    
    noise[i,:,:,0] = log_magnitude_noise.astype('float32')
    clean[i,:,:,0] = log_magnitude_clean.astype('float32')
    
    noise[i,:,:,1] = angle_noise.astype('float32')
    clean[i,:,:,1] = angle_clean.astype('float32')

#%%
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

#%%
noise_train = noise[:len(noise_list) - 40,:,:,:]
clean_train = clean[:len(clean_list) - 40,:,:,:]

noise_test = noise[len(noise_list) - 40:,:,:,:]
clean_test = clean[len(clean_list) - 40:,:,:,:]

train_dataset = tf.data.Dataset.from_tensor_slices((noise_train,clean_train))
print(train_dataset)
#train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

#%%
test_dataset = tf.data.Dataset.from_tensor_slices((noise_test,clean_test))
print(test_dataset)
# test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

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
for example_input, example_target in train_dataset.take(1):
  generate_images(generator, example_input, example_target)

#%%
EPOCHS = 100
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

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir {log_dir}

ls {checkpoint_dir}

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Run the trained model on a few examples from the test dataset
for inp, tar in test_dataset.take(10):
  generate_images(generator, inp, tar)

def unwrap_to_ang(unwrap):
    pi = 3.1415927410125732
    
    for i in range(1, int(np.max(unwrap) // pi + 2), 2):
        plus = (((i+2)*pi) >= unwrap) & (unwrap > (i*pi))
        minus = (unwrap >= (-(i+2)*pi)) & (unwrap < (-i*pi))
    
        unwrap[plus] -= (i+1)*pi
        unwrap[minus] += (i+1)*pi
    
    return unwrap

#%%
def mag_ang_to_stft(mag, ang):
    pi  = 3.1415927
    half = pi / 2
    
    over = ang > half
    lower = ang < -half
    plus = ((0 <= ang) & (ang <= half))
    minus = ((0 > ang) & (ang >= -half))
    
    ang_list = [plus, minus, over, lower]
    
    stft = np.zeros((256,256)).astype('complex64')
    

    for idx, i in enumerate(ang_list):
        
        real = mag[i]*np.cos(ang[i])
        imag = mag[i]*np.sin(ang[i])*complex(0,1)
        
        real_1 = mag[i]*np.cos(-ang[i])
        imag_1 = mag[i]*np.sin(-ang[i])*complex(0,1)
        
        real_2 = mag[i]*np.cos(pi - ang[i])
        imag_2 = mag[i]*np.sin(pi - ang[i])*complex(0,1)

        real_3 = mag[i]*np.cos(pi + ang[i])
        imag_3 = mag[i]*np.sin(pi + ang[i])*complex(0,1)
        
        if idx == 0:
            stft[i] = real + imag
        elif idx == 1:
            stft[i] = real_1 - imag_1
        elif idx == 2:
            stft[i] = -real_2 + imag_2
        elif idx == 3:
            stft[i] = -real_3 - imag_3
    
    return stft

#%% 모델이 예측해낸 magnitude와 angle을 기반으로 음성 추출
import soundfile

count = 560

for inp, tar in test_dataset.take(40):
  prediction = generator(inp, training = True)
  max_mag_clean = mag_max[count]
  min_mag_clean = mag_min[count]

  max_ang_clean = ang_max[count]
  min_ang_clean = ang_min[count]

  gen_mag = np.array(prediction[0][:,:,0])
  gt_mag = np.array(tar[0][:,:,0])

  gen_ang = np.array(prediction[0][:,:,1])
  gt_ang = np.array(tar[0][:,:,1])

  gen_mag_denorm = denormalization(gen_mag, -1, 1, max_mag_clean, min_mag_clean)
  gt_mag_denorm = denormalization(gt_mag, -1, 1, max_mag_clean, min_mag_clean)

  gen_ang_denorm = denormalization(gen_ang, -1, 1, max_ang_clean, min_ang_clean)
  gt_ang_denorm = denormalization(gt_ang, -1, 1, max_ang_clean, min_ang_clean)

  gen_db_amp = librosa.db_to_amplitude(gen_mag_denorm)
  gt_db_amp = librosa.db_to_amplitude(gt_mag_denorm)

  gen_unwrap_ang = unwrap_to_ang(gen_ang_denorm)
  gt_unwrap_ang = unwrap_to_ang(gt_ang_denorm)

  gen_stft = mag_ang_to_stft(gen_db_amp, gen_unwrap_ang)
  gt_stft = mag_ang_to_stft(gt_db_amp, gt_unwrap_ang)

  gen_wav = librosa.istft(gen_stft, hop_length = hop_length, win_length=n_fft)
  gt_wav = librosa.istft(gt_stft, hop_length = hop_length, win_length=n_fft)

  soundfile.write('/content/drive/MyDrive/gen_wav/gen_wav_{}.wav' .format(count), gen_wav, sr)
  soundfile.write('/content/drive/MyDrive/gt_wav/gt_wav_{}.wav' .format(count), gt_wav, sr)

  count += 1

#%% 모델이 예측해낸 magnitude만 가지고 librosa의 griffinlim을 이용하여 음성 추출
import soundfile

count = 560

for inp, tar in test_dataset.take(40):
  prediction = generator(inp, training = True)
  max_clean = mag_max[count]
  min_clean = mag_min[count]

  gen_mag = prediction[0][:,:,0]
  gt_mag = tar[0][:,:,0]

  gen_denorm = denormalization(gen_mag, -1, 1, max_clean, min_clean)
  gt_denorm = denormalization(gt_mag, -1, 1, max_clean, min_clean)

  gen_db_amp = librosa.db_to_amplitude(gen_denorm)
  gt_db_amp = librosa.db_to_amplitude(gt_denorm)

  # gen_wav = librosa.istft(gen_db_amp, hop_length = hop_length, win_length=n_fft)
  # gt_wav = librosa.istft(gt_db_amp, hop_length = hop_length, win_length=n_fft)

  gen_wav = librosa.griffinlim(gen_db_amp, hop_length = hop_length, win_length=n_fft)
  gt_wav = librosa.griffinlim(gt_db_amp, hop_length = hop_length, win_length=n_fft)

  soundfile.write('/content/drive/MyDrive/gen_wav_griffin/gen_wav_{}.wav' .format(count), gen_wav, sr)
#   soundfile.write('/content/drive/MyDrive/gt_wav/gt_wav_{}.wav' .format(count), gt_wav, sr)

  count += 1

plt.plot(gen_wav)
plt.show()

plt.plot(gt_wav)
plt.show()

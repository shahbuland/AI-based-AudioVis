import pygame

import numpy as np
from matplotlib import cm
import sounddevice as sd
import time

from utils import *

# This script creates an interactive visualization that lets user click a point and get associated audio

# Function that takes an tensor of a wf and plays it
def play_audio(wf, sr):
    wf = wf.numpy()
    sd.play(wf, sr, blocking = True)

# z : points in 2d space
# wfs : tensor of waveforms
# sr : sample rate
# labels : optional list of labels
# cmap : way of mapping numbers [0, 1] to RGB colors
def scatter_with_sounds(z, wfs, sr, labels = None, cmap = cm.get_cmap('hsv')):
    x = z[:,0]
    y = z[:,1]

    # Set up colors
    n_points = len(x)
    if labels is not None:
        labels = np.array(labels)
        labels += 1 # makes sure no labels get mapped to black (bg color)
        labels = labels / np.max(labels)
        c = np.array([cmap(l) for l in labels]) * 255
    else:
        c = [(255, 255, 255)] * n_points

    pygame.init()
    (w, h) = (800, 800)
    screen = pygame.display.set_mode((w, h))

    # Normalize points to lie within screen
    x = (x - x.min()) / (x.max() - x.min()) * w
    y = (y - y.min()) / (y.max() - y.min()) * h

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                # get closest point
                idx = np.argmin(np.sqrt(np.square(x - pos[0]) + np.square(y - pos[1])))
                play_audio(wfs[idx], sr)
        # draw points on  screen
        screen.fill((0, 0, 0))
        for i in range(len(x)):
            pygame.draw.circle(screen, c[i], (int(x[i]), int(y[i])), 5)
        pygame.display.flip()

# Suppose we have a decoder model that can decode a sequence of vectors into audio
# We want to create a visualization that lets us easily produce vector sequences
# decoder : has some method for converting vector sequence into waveform
# tform : some mapping from R^2 to latent space
# sr : how many vectors per second to collect
def draw_audio(decoder, tform, sr = 44100):
    # Set up pygame
    pygame.init()
    (w, h) = (800, 800)
    screen = pygame.display.set_mode((w, h))

    running = True
    recording = False

    path = []

    # Convert mouse path into sequence of vectors
    def path_to_vectors(path):
        path = np.stack(path, axis = 0)
        path = tform(path)

    # Using the mouse path, make sample and play it
    def play_path(path):
        vectors = path_to_vectors(path)
        wf = decoder(vectors)
        play_audio(wf, sr)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                recording = True
                path = []
            elif  event.type == pygame.MOUSEBUTTONUP:
                recording = False
                play_path(path)
                
            if recording:
                pos = pygame.mouse.get_pos()
                path.append(np.array([pos[0], pos[1]]))
        pygame.util.wait(10)
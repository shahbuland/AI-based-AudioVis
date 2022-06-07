import pygame

import numpy as np
from matplotlib import cm
import sounddevice as sd

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




def show_many_images(imgs, subtitles=(), patchess=(), title='', extents=(), cmaps=()):
    """Show many images at the same time!

    Look at implemention for details, or implement it yourself!!!

    Parameters
    ----------
    imgs: sequence of numpy array of shape (Mi, Ni) or (Mi, Ni, 3) or (Mi, Ni, 4)
    subtitles: sequence of str
        Suplots tiles
    patchess: sequence of sequence of matplotlib patches
        See `descartes` library
    title: str
        Window title
    extents: sequence of {(nbr, nbr, nbr, nbr) or None}
        The location, in data-coordinates, of the lower-left and upper-right corners.
        If using buzzard, pass fp.extent
    cmaps: sequence of {matplotlib's cmap or None}
        https://matplotlib.org/users/colormaps.html#grayscale-conversion

    Cursor modes
    ------------
    default: Multi cursor on multiplots
    o: zoom (toggle)
    p: move

    Shortcuts
    ---------
    h: home
    ctrl-w: close

    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import MultiCursor

    # Normalize input lengths
    imgs = list(imgs)
    count = len(imgs)

    subtitles = list(subtitles)
    subtitles = subtitles[:count]
    subtitles = subtitles + [''] * (count - len(subtitles))

    patchess = list(patchess)
    patchess = patchess[:count]
    patchess = patchess + [[]] * (count - len(patchess))

    extents = list(extents)
    extents = extents[:count]
    extents = extents + [(0, 1, 1, 0)] * (count - len(extents))

    cmaps = list(cmaps)
    cmaps = cmaps[:count]
    cmaps = cmaps + [None] * (count - len(cmaps))

    # # Normalize input image shapes
    # imgs_shape = np.vstack([img.shape[:2] for img in imgs]).max(0)
    # imgs = [
    #     ndi.zoom(img, shape, None, 0, 'nearest')
    #     for img in imgs
    #     for shape in [np.r_[imgs_shape / img.shape[:2], [1] * (len(img.shape) - 2)]]
    # ]

    # Decice layout
    if count == 1: w, h = 1, 1
    elif count == 2: w, h = 2, 1
    elif count in {3, 4}: w, h = 2, 2
    elif count in {5, 6}: w, h = 3, 2
    elif count in {7, 8, 9}: w, h = 3, 3
    elif count in {10, 11, 12}: w, h = 4, 3
    elif count in {13, 14, 15, 16}: w, h = 4, 4
    else: assert False

    # Plot images and patches
    fig = plt.figure()
    axes = []
    for i, img, subtitle, patches, extent, cmap in zip(range(count), imgs, subtitles, patchess, extents, cmaps):
        if i == 0:
            a = fig.add_subplot(h, w, i + 1)
        else:
            a = fig.add_subplot(h, w, i + 1, sharex=axes[0], sharey=axes[0])
        axes.append(a)
        plt.imshow(img, cmap=cmap, extent=extent)
        a.set_title(subtitle)
        for patch in patches:
            a.add_patch(patch)

    # Misc
    multi = MultiCursor(fig.canvas, axes, color='black', lw=0.5, horizOn=True, vertOn=True)
    fig.canvas.set_window_title(title)
    fig.subplots_adjust(
        wspace=0.10, hspace=0.11,
        left=0.04, right=1. - 0.01,
        bottom=0.03, top=1. - 0.03,
    )

    # Set fullscreen
    manager = plt.get_current_fig_manager()
    if hasattr(manager, 'full_screen_toggle'):
        manager.full_screen_toggle()
    elif hasattr(manager, 'window'):
        manager.window.showMaximized()
    elif hasattr(manager, 'frame'):
        manager.frame.Maximized(True)
    else:
        print('how to full screen?')

    # Fire
    plt.show()

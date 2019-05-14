import sys, os, multiprocessing

multiprocessing.freeze_support()

from soundfile import read as sfread
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import k_means
from librosa.feature import mfcc
from fire import Fire
from tqdm import tqdm

def read_audio(fname):
    audio, samplerate = sfread(fname)
    if audio.ndim>1:
        audio = audio.mean(1)
    return audio, samplerate

def write_obj(fname, vertices, lines):
    with open(fname, 'w') as file:
        for v in vertices:
            file.write('v '+' '.join(str(el) for el in v)+'\n')
        for l in lines:
            file.write('l '+' '.join(str(int(el)) for el in l)+'\n')

def plot_vertices(ax, v, s=9):
    idxs = np.argsort(v[:, 2])
    ax.scatter(v[idxs, 0], v[idxs, 1], c=-v[idxs, 2], s=s, cmap='YlOrRd')

def plot_glyph(fname, vertices, height=600):
    fig, ax = plt.subplots(1,3,figsize=(height/100*3, height/100))
    for a in ax:
        a.axis('off')
    s = (height/200)**2
    plot_vertices(ax[0], vertices, s)
    plot_vertices(ax[1], vertices[:, [1,2,0]], s)
    plot_vertices(ax[2], vertices[:, [2,0,1]], s)
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(fname, facecolor='black')
    plt.close(fig)

def project_ortho(theta0, phi0, theta1, phi1):
    v0 = pol2car(theta0, phi0)
    v1 = pol2car(theta1, phi1)
    v2 = v1-np.dot(v0, v1)*v0
    v2 /= np.linalg.norm(v2)
    return car2pol(*v2)

def pol2car(theta, phi):
    x,y = np.cos(theta), np.sin(theta)
    h,z = np.cos(phi), np.sin(phi)
    return np.array([x*h, y*h, z])

def car2pol(x, y, z):
    phi = np.arcsin(z)
    theta = np.arctan2(y, x)
    return theta, phi

def glyph(audio, samplerate, steps=1000, lp_coef=.99, twist_mult=1):
    mfcc_spect = mfcc(audio, samplerate, n_mfcc=26)
    n = int(mfcc_spect.std(1).mean(0)/8)+1
    centroids, _, _ = k_means(
        mfcc_spect[1:].T, n,
        init=mfcc_spect[1:,np.linspace(0, mfcc_spect.shape[1]-1, n).astype(np.int64)].T,
        n_init=1)
    centroids -= centroids.mean(0)
    centroids /= centroids.std(0)
    centroids = centroids[np.argsort(centroids[:, 0])] # deterministic order

    segments = []
    vertices, lines = [], []
    for i,centroid in enumerate(centroids):
        x0, y0, z0, theta0, phi0 = centroid[:5]
        theta0 *= np.pi/2
        phi0 *= np.pi/4
        fb = list(reversed(centroid[5:]))
        length = 1/2
        twist = twist_mult/2
        if i>0:
            branch = (x0/(1+abs(x0))*.5+.5) * len(segments)
            branch_on, branch_pt = int(branch), branch%1
            branch_idx = int(len(segments[branch_on])*branch_pt)
            length = (y0/(1+abs(y0))*.5+.5)**2
            twist = z0*abs(z0)**.5*twist_mult*2
            x0, y0, z0, theta_ortho, phi_ortho = segments[branch_on][branch_idx, :5]
            theta0, phi0 = project_ortho(theta_ortho, phi_ortho, theta0, phi0)
            lines.append([sum(len(s) for s in segments[:branch_on])+branch_idx+1])
        else:
            vertices.append(np.array([x0, y0, z0]))
            lines.append([1])
        twist/=2
        s = [np.array([x0, y0, z0, theta0, phi0])]

        z = s[0].copy()
        for i in range(steps//8+int(length*steps)):
            pt = s[-1].copy()
            z = z*lp_coef + pt*(1-lp_coef)
            pt[3] += (
                fb[18] +
                fb[0]*z[0] + fb[2]*z[1] + fb[4]*z[2] +
                fb[6]*z[0]*z[1] + fb[8]*z[1]*z[2] + fb[10]*z[2]*z[0] +
                fb[12]*z[0]**2 + fb[14]*z[1]**2 + fb[16]*z[2]**2
            )*np.pi/steps*twist
            pt[4] += (
                fb[19] +
                fb[1]*z[0] + fb[3]*z[1] + fb[5]*z[2] +
                fb[7]*z[0]*z[1] + fb[9]*z[1]*z[2] + fb[11]*z[2]*z[0] +
                fb[13]*z[0]**2 + fb[15]*z[1]**2 + fb[17]*z[2]**2
            )*np.pi/steps*twist
            pt[:3] += pol2car(pt[3], pt[4])/steps

            s.append(pt)
            vertices.append(pt[:3])
            lines[-1].append(len(vertices))


        segments.append(np.array(s))

    vertices = np.array(vertices)
    vertices -= vertices.mean(0)

    return vertices, lines

def convert_path(path, new_dir, new_ext):
    return os.path.join(new_dir,
        '.'.join(os.path.split(path)[1].split('.')[:-1]+[new_ext]))

def main(audio_dir, obj_dir=None, image_dir=None, image_height=600,
        num_points=1000, twist=1
    ):
    """Generate 3d glyphs from audio clips.

    Args:
        audio_dir: directory containing clips as .wav(e) and .aif(f) files
        obj_dir: directory to output .obj files containing lines (optional)
        image_dir: directory to output .png previews (optional)
        image_height: vertical image dimension in pixels (default 600)
        num_points: approximate number of vertices per glyph (default 1000)
        twist: twistiness parameter for glyphs (default 1)
    """
    if obj_dir is None and image_dir is None:
        raise ValueError('no output specified; use --obj-dir and/or --image-dir')

    audio_names = [
        os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
        if '.wav' in f or '.aif' in f]

    if not audio_names:
        raise ValueError(
            f'no .wav(e) or .aif(f) files found in --audio-dir  "{audio_dir}"')

    matplotlib.use('agg')

    for audio_name in tqdm(audio_names, desc='processing files'):
        vertices, lines = glyph(
            *read_audio(audio_name),
            twist_mult=twist, steps=num_points)
        if obj_dir:
            if not os.path.exists(obj_dir):
                os.mkdir(obj_dir)
            obj_name = convert_path(audio_name, obj_dir, 'obj')
            write_obj(obj_name, vertices, lines)
        if image_dir:
            if not os.path.exists(image_dir):
                os.mkdir(image_dir)
            image_name = convert_path(audio_name, image_dir, 'png')
            plot_glyph(image_name, vertices, height=image_height)

if __name__=='__main__':
    Fire(main)

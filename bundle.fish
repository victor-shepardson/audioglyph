set sndfile_lib (dirname (which python))/../lib/python3.7/site-packages/_soundfile_data/libsndfile.dylib

pyinstaller audioglyph.py -y --clean\
    --add-binary $sndfile_lib:_soundfile_data \
    --hidden-import sklearn.neighbors.typedefs \
    --hidden-import sklearn.neighbors.quad_tree \
    --hidden-import sklearn.tree --hidden-import sklearn.tree._utils

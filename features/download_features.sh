wget --recursive --no-parent -l1 -nH --cut-dirs=4 --reject="index.html*" --user $1 --password $2 https://thor.robots.ox.ac.uk/~vgg/data/bobsl/features/video-swin-s_c8697_16f_bs32/
cd video-swin-s_c8697_16f_bs32
for g in *.gz; do gunzip $g; done
cd ..
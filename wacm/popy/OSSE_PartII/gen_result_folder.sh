#/bin/bash

#ln -s result.benchmark0/rst_ms/2012060703_rst.nc result/rst_ms/2012060703_rst.nc
#ln -s result.benchmark0/avg_ms/2012060703_avg.nc result/avg_ms/2012060703_avg.nc

#$1 e.g. result.benchmark008
#mkdir -p $1

#cd $1
mkdir result
cd result

mkdir -p analysis_lr analysis_ms avg_month  avg_noda his_ms_bias  rst_lr rst_ms rst_noda analysis_lr_bias  analysis_ms_bias avg_ms  his_ms his_noda rst_lr_bias  rst_ms_bias

ln -s ../result.001/forcing .
cp -p ../result.001/avg_ms/2012060603_avg.nc ./avg_ms/
cp -p ../result.001/rst_ms/2012060703_rst.nc ./rst_ms/

ln -s ../result.001/clim ./
ln -s ../result.001/grid ./
ln -s ../result.001/remote_* ./
ln -s ../result.001/insitu_* ./

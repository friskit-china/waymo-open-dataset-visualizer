# https://unix.stackexchange.com/questions/28803/how-can-i-reduce-a-videos-size-with-ffmpeg


PREFIX=620 # 620, 767
FLAG="-c:v libx264 -pix_fmt yuv420p -vf scale=-1:720 -b:v 1600k"

echo ffmpeg -framerate 10 -pattern_type glob -i "\"$PREFIX.output/camera-label-front/*.png\""       $FLAG camera-label-front.$PREFIX.mp4 >> run.sh
echo ffmpeg -framerate 10 -pattern_type glob -i "\"$PREFIX.output/camera-label-front-left/*.png\""  $FLAG camera-label-front-left.$PREFIX.mp4 >> run.sh
echo ffmpeg -framerate 10 -pattern_type glob -i "\"$PREFIX.output/camera-label-front-right/*.png\"" $FLAG camera-label-front-right.$PREFIX.mp4 >> run.sh
echo ffmpeg -framerate 10 -pattern_type glob -i "\"$PREFIX.output/camera-label-side-left/*.png\""   $FLAG camera-label-side-left.$PREFIX.mp4 >> run.sh
echo ffmpeg -framerate 10 -pattern_type glob -i "\"$PREFIX.output/camera-label-side-right/*.png\""  $FLAG camera-label-side-right.$PREFIX.mp4 >> run.sh
echo ffmpeg -framerate 10 -pattern_type glob -i "\"$PREFIX.output/camera-ori-front/*.png\""         $FLAG camera-ori-front.$PREFIX.mp4 >> run.sh
echo ffmpeg -framerate 10 -pattern_type glob -i "\"$PREFIX.output/camera-ori-front-left/*.png\""    $FLAG camera-ori-front-left.$PREFIX.mp4 >> run.sh
echo ffmpeg -framerate 10 -pattern_type glob -i "\"$PREFIX.output/camera-ori-front-right/*.png\""   $FLAG camera-ori-front-right.$PREFIX.mp4 >> run.sh
echo ffmpeg -framerate 10 -pattern_type glob -i "\"$PREFIX.output/camera-ori-side-left/*.png\""     $FLAG camera-ori-side-left.$PREFIX.mp4 >> run.sh
echo ffmpeg -framerate 10 -pattern_type glob -i "\"$PREFIX.output/camera-ori-side-right/*.png\""    $FLAG camera-ori-side-right.$PREFIX.mp4 >> run.sh
echo ffmpeg -framerate 10 -pattern_type glob -i "\"$PREFIX.output/lidar-bev-label/*.png\""          $FLAG lidar-bev-label.$PREFIX.mp4 >> run.sh
echo ffmpeg -framerate 10 -pattern_type glob -i "\"$PREFIX.output/lidar-bev-ori/*.png\""            $FLAG lidar-bev-ori.$PREFIX.mp4 >> run.sh
echo ffmpeg -framerate 10 -pattern_type glob -i "\"$PREFIX.output/lidar-normal-label/*.png\""       $FLAG lidar-normal-label.$PREFIX.mp4 >> run.sh
echo ffmpeg -framerate 10 -pattern_type glob -i "\"$PREFIX.output/lidar-normal-ori/*.png\""         $FLAG lidar-normal-ori.$PREFIX.mp4 >> run.sh
echo ffmpeg -framerate 10 -pattern_type glob -i "\"$PREFIX.output/range-image-front/*.png\""        $FLAG range-image-front.$PREFIX.mp4 >> run.sh
echo rm run.sh >> run.sh
chmod +x run.sh
./run.sh
# rm run.sh >> /dev/null

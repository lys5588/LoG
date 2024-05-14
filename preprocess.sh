data=./preprocess/Yuehai
#withoutgps
#colmap feature_extractor --database_path ${data}/database.db --image_path ${data}/images --ImageReader.camera_model OPENCV --ImageReader.single_camera_per_folder 1 --SiftExtraction.use_gpu 1
#colmap vocab_tree_matcher --database_path ${data}/database.db --VocabTreeMatching.vocab_tree_path ./vocab_tree_flickr100K_words1M.bin --VocabTreeMatching.num_images 100 --SiftMatching.use_gpu 1
#mkdir ${data}/sparse
#colmap mapper --database_path ${data}/database.db --image_path ${data}/images --output_path ${data}/sparse
#python3 apps/calibration/align_with_cam.py --colmap_path ${data}/sparse/0 --target_path ${data}/sparse_align
#python3 apps/calibration/read_colmap.py ${data}/sparse_align --min_views 3
#python3 apps/test_dataset.py --cfg config/example/Hospital/dataset.yml split dataset

#withGPS
colmap feature_extractor --database_path ${data}/database.db --image_path ${data}/images --ImageReader.camera_model OPENCV --ImageReader.single_camera_per_folder 1 --SiftExtraction.use_gpu 1
# matching use GPS info from images, max_distance=300
colmap spatial_matcher --database_path ${data}/database.db --SpatialMatching.max_num_neighbors 200 --SpatialMatching.max_distance 300 --SiftMatching.use_gpu 1
mkdir ${data}/sparse
colmap mapper --database_path ${data}/database.db --image_path ${data}/images --output_path ${data}/sparse
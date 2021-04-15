
# Tasks

- [X] Phantoscope default
- [ ] Phantoscope with gpu
- [ ] 
Fix  bug : mysql container failed to start
https://qiita.com/AK4747471/items/5e82e6b776762412a3b8

We also need to remove this dir /mnt/om

## Phantoscope default

Follow : [Phantoscope Quick Start](https://github.com/zilliztech/phantoscope/tree/0.1.0/docs/site/en/quickstart)

```bash
export LOCAL_ADDRESS=$(ip a | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'| head -n 1)
docker-compose up -d
docker-compose ps

```

## Phantoscope  Detect object from an Image and Search




## Phantoscope : gpu image

```bash
# minvus gpu
https://www.milvus.io/docs/guides/get_started/install_milvus/gpu_milvus_docker.md

# install nvidia-docker
https://github.com/NVIDIA/nvidia-docker

# then run minvus gpu
docker pull milvusdb/milvus:0.10.0-gpu-d061620-5f3c00
mkdir -p /home/$USER/opt/milvus/conf
cd /home/$USER/opt/milvus/conf
wget https://raw.githubusercontent.com/milvus-io/milvus/v0.10.0/core/conf/demo/server_config.yaml


docker run -d --name milvus_gpu_0.10.0 --gpus 0 \
-p 19530:19530 \
-p 19121:19121 \
-v /home/$USER/opt/milvus/db:/var/lib/milvus/db \
-v /home/$USER/opt/milvus/conf:/var/lib/milvus/conf \
-v /home/$USER/opt/milvus/logs:/var/lib/milvus/logs \
-v /home/$USER/opt/milvus/wal:/var/lib/milvus/wal \
milvusdb/milvus:0.10.0-gpu-d061620-5f3c00

```
## Phantoscope : ...










## Reference

1. [milvus install from source](https://github.com/milvus-io/milvus/blob/master/INSTALL.md)
2. [milvus tool sizing](https://milvus.io/tools/sizing)








[{"_id": "1594122127727937000", "_app_name": "object-example", 
"_image_url": "http://192.168.0.106:9000/object-s3/object-example-994e1052ed5e488dacb1e5283480ff3a", "_fields": {"object_field": {"type": "object", "pipeline": "object_pipeline"}}}, {"_id": "1594122204748269000", "_app_name": "object-example", "_image_url": "http://192.168.0.106:9000/object-s3/object-example-cfd04e456376470f8cc30f7ddc301aed", "_fields": {"object_field": {"type": "object", "pipeline": "object_pipeline"}}}, {"_id": "1594122204748269001", "_app_name": "object-example", "_image_url": "http://192.168.0.106:9000/object-s3/object-example-cfd04e456376470f8cc30f7ddc301aed", "_fields": {"object_field": {"type": "object", "pipeline": "object_pipeline"}}}, {"_id": "1594122289279742000", "_app_name": "object-example", "_image_url": "http://192.168.0.106:9000/object-s3/object-example-123991d970d647acbe84115998689295", "_fields": {"object_field": {"type": "object", "pipeline": "object_pipeline"}}}, {"_id": "1594122308332965000", "_app_name": "object-example", "_image_url": "http://192.168.0.106:9000/object-s3/object-example-3c9d133cfb2042bd942cd895e5a6478a", "_fields": {"object_field": {"type": "object", "pipeline": "object_pipeline"}}}]


https://img.webmd.com/dtmcms/live/webmd/consumer_assets/site_images/article_thumbnails/slideshows/surprises_about_dogs_and_cats_slideshow/1800x1200_surprises_about_dogs_and_cats_slideshow.jpg

curl --location --request POST ${LOCAL_ADDRESS}':5000/v1/application/object-example/search' \
--header 'Content-Type: application/json' \
--data '{
	"fields": {
        "object_field": {
            "url": "https://i.insider.com/536aa78069bedddb13c60c3a?width=1100&format=jpeg&auto=webp"
        }
    },
    "topk": 1
}'
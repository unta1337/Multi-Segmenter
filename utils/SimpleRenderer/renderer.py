import bpy
import os
import math
import shutil
import argparse
import imageio.v2 as imageio

# parser 객체 생성
parser = argparse.ArgumentParser()

# 인자 추가
parser.add_argument("--fps", type=int, default=60, help="FPS argument")
parser.add_argument("--frame", type=int, default=5 ,help="Frame argument")
parser.add_argument("--quality", type=int, default=6, help="Quality argument")
parser.add_argument("--align", action="store_true", help="Align argument")
parser.add_argument("--videoonly", action="store_true", help="Align argument")
parser.add_argument("filepath", help="Filepath argument")

# 인자 파싱
args = parser.parse_args()

fps = args.fps
frame = args.frame
quality = args.quality
align = args.align
videoonly = args.videoonly
filepath = args.filepath

dir_path = os.path.dirname(filepath)
file_name, file_ext = os.path.splitext(os.path.basename(filepath))

# 작업 디렉토리 설정
directory = dir_path
os.chdir(directory)

# 출력 디렉토리 설정
output_dir = os.path.abspath("./frames/")
os.makedirs(output_dir, exist_ok=True) # 디렉토리가 없으면 생성

# 모든 오브젝트 가져오기
objects = bpy.context.selected_objects

# Cube 객체 제거하기
for obj in objects:
    if obj.name == "Cube":
        bpy.data.objects.remove(obj, do_unlink=True)

# obj 파일 가져 오기
bpy.ops.import_scene.obj(filepath=filepath)

# 뷰어 설정 및 렌더링
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.image_settings.file_format = 'PNG'

# 모든 오브젝트 가져오기
objects = bpy.context.selected_objects

# 모든 오브젝트를 Collection에 넣기
collection_name = "AllObjects"
if not bpy.data.collections.get(collection_name):
    bpy.data.collections.new(collection_name)
bpy.context.scene.collection.children.link(bpy.data.collections[collection_name])
for obj in objects:
    bpy.data.collections[collection_name].objects.link(obj)

# 모든 오브젝트를 선택
for obj in bpy.context.scene.objects:
    obj.select_set(True)

# 카메라를 선택한 상태에서 선택한 오브젝트가 모두 보이도록 이동
bpy.ops.view3d.camera_to_view_selected()

# 회전 애니메이션 만들기
axes = ['x', 'y', 'z'];
for axis in range(3):
    for obj in objects:
        obj.rotation_euler = (0, 0, 0)

    for i in range(frame):
        filepath = os.path.join(output_dir, f"{file_name}_{axes[axis]}_{i:04d}.png")
        bpy.context.scene.render.filepath = filepath

        # 모든 오브젝트 회전 설정
        for obj in objects:
            obj.rotation_euler[axis] = i * 2 * math.pi / frame # z축 기준으로 회전

        if align:
            # 모든 오브젝트를 선택
            for obj in bpy.context.scene.objects:
                obj.select_set(True)

            # 카메라를 선택한 상태에서 선택한 오브젝트가 모두 보이도록 이동
            bpy.ops.view3d.camera_to_view_selected()

        # 렌더링 실행
        bpy.ops.render.render(write_still=True)

for axis in axes:
    writer = imageio.get_writer(f"{dir_path}/{file_name}_{axis}_axis.mp4", fps=fps, quality=quality)
    for i in range(frame):
        file = os.path.join(output_dir, f"{file_name}_{axis}_{i:04d}.png")
        im = imageio.imread(file)
        writer.append_data(im)
    writer.close()

if videoonly:
    shutil.rmtree(output_dir)
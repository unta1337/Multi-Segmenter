import bpy
import os
import math
import shutil
import argparse
import imageio.v2 as imageio

# parser 객체 생성
parser = argparse.ArgumentParser()

# 인자 추가
parser.add_argument("--fps", type=int, default=10, help="FPS argument")
parser.add_argument("--frame", type=int, default=60, help="Frame argument")
parser.add_argument("--quality", type=int, default=6, help="Quality argument")
parser.add_argument("--align", action="store_true", help="Align argument")
parser.add_argument("--videoonly", action="store_true", help="Video only argument")
parser.add_argument("--scale", type=int, default=100, help="Scale argument")
parser.add_argument("filepath", help="Filepath argument")

# 인자 파싱
args = parser.parse_args()

fps = args.fps
frame = args.frame
quality = args.quality
align = args.align
videoonly = args.videoonly
filepath = os.path.abspath(args.filepath)
scale = args.scale

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

# 기본 객체 제거하기
for obj in objects:
    bpy.data.objects.remove(obj, do_unlink=True)

# obj 파일 가져 오기
bpy.ops.import_scene.obj(filepath=filepath)

# 뷰어 설정 및 렌더링
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.resolution_percentage = scale

# 모든 오브젝트 가져오기
objects = bpy.context.selected_objects

# 카메라를 원점(0, 0, 0)을 중심으로 회전시키기 위해 새로운 빈 객체(empty object) 생성
bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0))

# 카메라를 빈 객체에 부착하고 위치 조정
camera = bpy.data.objects['Camera']
camera.parent = bpy.context.active_object

# 전역 광원을 장면에 추가
bpy.ops.object.light_add(type='SUN', align='WORLD', location=(0, 0, 10))
sun_light = bpy.context.active_object

# 전역 광원 설정
sun_light.data.energy = 1
sun_light.data.specular_factor = 0.5
sun_light.rotation_euler = (math.radians(90), 0, 0)

# 모든 오브젝트를 선택
for obj in bpy.context.scene.objects:
    obj.select_set(True)

# 카메라를 선택한 상태에서 선택한 오브젝트가 모두 보이도록 이동
bpy.ops.view3d.camera_to_view_selected()

# 회전 애니메이션 만들기
axes = ['x', 'y', 'z'];
for axis in range(3):
    camera.parent.rotation_euler = [0, 0, 0]

    for i in range(frame):
        filepath = os.path.join(output_dir, f"{file_name}_{axes[axis]}_{i:04d}.png")
        bpy.context.scene.render.filepath = filepath

        # 카메라가 객체를 중심으로 회전
        camera.parent.rotation_euler[axis] = i * 2 * math.pi / frame

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
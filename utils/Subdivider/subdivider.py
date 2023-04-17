import bpy
import os
import argparse

# parser 객체 생성
parser = argparse.ArgumentParser()

# 인자 추가
parser.add_argument("--cuts", type=int, default=1, help="Cuts argument")
parser.add_argument("--min", type=int, default=0, help="Min argument")
parser.add_argument("--max", type=int, default=0, help="Max argument")
parser.add_argument("filepath", help="Filepath argument")

# 인자 파싱
args = parser.parse_args()

cuts = args.cuts
min = args.min
max = args.max + 1
if min == 0 and max == 1:
    min = cuts
    max = cuts + 1

filepath = os.path.abspath(args.filepath)

dir_path = os.path.dirname(filepath)
file_name, file_ext = os.path.splitext(os.path.basename(filepath))

# 작업 디렉토리 설정
directory = dir_path
os.chdir(directory)


def load_obj(filepath):
    # 기존 오브젝트 삭제
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # OBJ 파일 로드
    bpy.ops.import_scene.obj(filepath=filepath)


# Subdivide 함수
def apply_subdivide(cuts):
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.subdivide(number_cuts=cuts)
            bpy.ops.object.mode_set(mode='OBJECT')


# OBJ 저장 함수
def save_obj(filepath):
    bpy.ops.export_scene.obj(filepath=filepath, use_selection=True)


# OBJ 파일 경로
source_obj_path = filepath

# 저장할 경로 설정
output_folder = dir_path
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for c in range(min, max):
    # OBJ 파일 로드
    load_obj(source_obj_path)
    apply_subdivide(c)
    # 파일명에 subdivide 횟수를 포함시킴
    output_file_name = f"{file_name}_subdivided_{c}{file_ext}"
    output_file_path = os.path.join(output_folder, output_file_name)
    save_obj(output_file_path)

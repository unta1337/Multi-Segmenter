# Multi-Segmentor
한국기술교육대학교 2023학년도 1학기 멀티코어프로그래밍 텀프로젝트

본 프로그램은 3차원 물체가 저장된 OBJ 파일과 물체 재질 정보가 저장된 MTL
파일을 입력으로 받아, 물체의 각 부분이 구분된 OBJ, MTL 파일과 성능 측정 결과를
TXT 파일로 출력하는 명령줄 프로그램이다.

# 개요
![V1](https://github.com/unta1337/Multi-Segmenter/assets/58779799/24769344-bcfc-43e7-85d5-dadd8ba60cf2)

명령줄 인자를 통해 프로그램의 실행 제어가 가능하며, 지원하는 명령줄 인자는
다음과 같다.
- Mode: “serial”, “parallel” 또는 “cuda” 값을 가질 수 있으며, 직렬
알고리즘을 사용할 지, 병렬 알고리즘을 사용할 지 결정한다.
- Tolerance (Optional): 물체 분할 전처리 과정에서 법선 벡터 병합이
이루어지는 기준 각도이다. 생략 가능하며 기본 값으로 15도를 가진다.
- ObjFilePath: 분할할 OBJ 파일이 위치한 경로이다. 동일한 경로에
존재하는 이름이 같은 MTL 파일도 동시에 읽어와 처리한다.
실행된 프로그램은 ObjFilePath와 같은 디렉터리에, 분할이 완료된 OBJ, MTL과 성능
측정 결과 파일을 “Segmented_[Mode]\_[Tolerance]\_[FileName].{obj,mtl,txt}”의 형태로
저장한다.

![V2](https://github.com/unta1337/Multi-Segmenter/assets/58779799/84b4470d-4247-4f0a-a96b-1b69b90ab194)  

 > 올바른 명령줄 인자 입력으로, 제대로 실행된 프로그램의 모습

![V3](https://github.com/unta1337/Multi-Segmenter/assets/58779799/fdd9e115-a882-400f-9c6b-74895b0f76c3)
 > 프로그램의 출력이 저장된 모습

![V4](https://github.com/unta1337/Multi-Segmenter/assets/58779799/608f434e-3b11-4734-9fda-656334697325)  
 > 성능 측정 결과 파일의 모습 (Segmented_serial_15.0_TestCube.txt)

![V7](https://github.com/unta1337/Multi-Segmenter/assets/58779799/f26aa092-2800-4c25-9c51-84d2438f65aa)  
 > 분할된 OBJ 파일을 3차원 뷰어 프로그램에서 확인한 모습

분할된 OBJ, MTL 파일은 온라인 3D 뷰어(3dviewer.net)에서 파일을 드래그 앤 드롭으로 불러와 확인하거나, 오픈소스 3D 컴퓨터 그래픽 소프트웨어인 Blender에서 “Import - Wavefront (.obj)” 메뉴를 이용하여 확인할 수 있다.

# 직렬 알고리즘
직렬 알고리즘은 3D 모델의 폴리곤(삼각형)을 처리하기 위해 "법선 벡터 그룹화"와 "연결성 검사 및 재분획"의 두 단계로 구성된다.

## 법선 벡터 그룹화
![G1](https://github.com/unta1337/Multi-Segmenter/assets/58779799/068a3a55-fa49-41f3-bc1c-96583bc05b71)
 - 3D 모델의 각 폴리곤의 법선(방향) 벡터를 계산한다.
 - 주어진 'tolerance' 각도 값을 기반으로 비슷한 방향의 법선 벡터를 그룹화한다.
 - 이 단계에서 법선 벡터와 연관된 삼각형 폴리곤을 자료구조에 저장한다.

## 연결성 검사 및 재분획
![G2](https://github.com/unta1337/Multi-Segmenter/assets/58779799/619a1ec0-5f49-46b1-93bb-e546bdd7a583)
![CC](https://github.com/unta1337/Multi-Segmenter/assets/58779799/62436e49-2959-42a2-9866-63b294022cec)

 - 폴리곤의 인접 관계를 분석한다.
 - 인접한 폴리곤을 찾기 위해 DFS(깊이 우선 탐색)를 사용한다.
 - 인접 관계에 따라 원래 법선 벡터 그룹을 서브 그룹으로 나눈다.
 - 이 알고리즘은 효율성을 위해 재귀 호출이 아닌 스택을 사용하여 DFS를 구현하고, 각 폴리곤에 그룹 번호를 할당하여 최종 그룹을 형성한다. 이 과정을 통해 3D 모델을 효과적으로 세분화할 수 있다.

# 병렬 알고리즘 (OpenMP)
OpenMP 병렬 알고리즘은 크게 "Segmenter"와 "FaceGraph" 두 부분으로 나누어져 있으며, OBJ 파일로부터 폴리곤(다각형)과 법선 벡터(면의 방향을 나타내는 벡터) 정보를 처리한다.
## Segmenter
 - OBJ 파일을 읽고, 폴리곤을 법선 벡터 기준으로 분류한다.
 - OpenMP를 사용하여 병렬 처리를 수행한다.
![P1](https://github.com/unta1337/Multi-Segmenter/assets/58779799/dc8f13fb-2577-4f8f-aa01-79c4da46ea1c)
 - "tolerance" 값에 따라 법선 벡터를 그룹화하되, 병렬 처리 시 "half_tolerance"를 사용해 오차를 줄인다.

## FaceGraph
 - Segmenter에서 생성된 법선 벡터 그룹을 다룬다.
 - 인접한 폴리곤을 기반으로 세그먼트를 생성한다.
 - 병렬 처리를 효율적으로 하기 위한 여러 전략이 적용되어 있다.

## 동기화
![](https://github.com/unta1337/Multi-Segmenter/assets/58779799/622a56f5-eff0-4e5d-83f0-e2bb4cf2de1f)
병렬 처리에서 발생할 수 있는 문제를 피하기 위해 각 법선 벡터별로 별도의 lock을 사용한다.
lock을 저장하기 위한 자료구조로 STL의 unordered_map 대신 병렬 처리에 적합한 자료구조를 사용한다. 여기서는 Open Addressing을 사용한 해시맵이 사용된다.
최종적으로, 인접한 삼각형 그룹을 분류하고 이를 시각화할 수 있는 정보로 변환한다. 여기에도 동기화가 필요하며, 이는 OpenMP의 "omp critical"을 통해 해결한다.

# 병렬 알고리즘 (CUDA + OpenMP)
CUDA (GPU)와 OpenMP (CPU)를 이용하여 만든 이기종 컴퓨팅 기법이 적용된 병렬 알고리즘이다.  모델을 구성하는 삼각형의 법선 벡터를 기준으로 그룹화하는 과정들을 GPU 상에셔 연산하는 방식을 사용한다.

## 법선 벡터 계산
![L2](https://github.com/unta1337/Multi-Segmenter/assets/58779799/092da8e0-fb31-40c4-acc7-c4f102afee3c)  
각 삼각형의 법선 벡터를 계산한다.
### 법선 벡터의 각도 계산
![L3](https://github.com/unta1337/Multi-Segmenter/assets/58779799/9a6990ce-480a-4056-856d-17fb59b39541)  
각 법선 벡터를 x, y, z 축에 투영하여 각도를 계산한다.  

### 그룹 ID 반환
각도를 기준으로 삼각형을 여러 그룹으로 분류하고, 각 삼각형에 그룹 ID를 할당한다.

### 삼각형 정렬
그룹 ID를 기준으로 삼각형을 정렬한다. 그 후, 각 그룹의 시작 위치와 그룹에 속한 삼각형의 수를 구한다.
이 과정을 통해 병렬 연산에 적합한 구조를 만들어, 전체 연산 과정을 효율적으로 만든다.

### 삼각형의 분류
![L4](https://github.com/unta1337/Multi-Segmenter/assets/58779799/031c62d0-20b0-4040-98f6-597aea18e967)  
삼각형은 먼저 법선(노멀)의 방향에 따라 분류된다.

### 서브그룹 분류
![L5](https://github.com/unta1337/Multi-Segmenter/assets/58779799/e8b9fe67-3da6-4979-b1e4-2fa8a0096af6)  
각 그룹 안의 삼각형은 연결 여부에 따라 다시 서브그룹으로 나뉜다.

### 정점에 대한 삼각형 인접 리스트 생성
![L6](https://github.com/unta1337/Multi-Segmenter/assets/58779799/cf80b466-31a1-440e-a847-49ab17eed1c7)  
각 정점에 연결된 삼각형의 정보를 생성한다. 해시 맵 대신 룩업 테이블을 사용하여 효율성을 높인다. CUDA를 사용하여 병렬 처리를 적용한다.

### 삼각형에 대한 삼각형 인접 리스트 생성
![L8](https://github.com/unta1337/Multi-Segmenter/assets/58779799/9f8b6cc3-3c59-4046-91e9-9263db66a808)  
각 삼각형이 어떤 다른 삼각형과 연결되어 있는지를 파악한다. 이 정보는 다시 정점에 대한 인접 리스트를 통해 얻는다.
이 과정은 주로 GPU에서 병렬 처리를 통해 수행되며, 아토믹 연산과 같은 동기화 기법을 사용하여 멀티 쓰레딩 문제를 해결한다. 마지막에는 각 블록(쓰레드 그룹)에서 처리한 정보를 취합하여 최종 인접 리스트를 완성한다.

## 그룹 번호 부여
법선 벡터의 방향과 연결성을 기반으로 분류된 삼각형을 그룹화한다. 이 과정에서 유니온-파인드 데이터 구조가 사용되는데, CUDA에서는 이를 변형한 룩업 테이블과 선형 룩업 테이블을 이용한다.
![L10](https://github.com/unta1337/Multi-Segmenter/assets/58779799/50cdb1e4-679f-43e9-ac8f-d3b7c999fb25)

## 정점 번호 부여
그룹화된 삼각형의 각 정점에 고유 번호를 부여한다. 직렬 알고리즘과 병렬 CUDA 구현 모두에서 이 과정이 필요하며, CUDA에서는 모든 그룹을 동시에 처리한다.  
![L12](https://github.com/unta1337/Multi-Segmenter/assets/58779799/c60d6568-52b7-4d84-bfe8-a0e1d39d11f7)

## 메모리 관리와 성능 최적화
CUDA 스트림을 이용하여 비동기적으로 메모리 할당과 초기화를 수행한다.

## 멀티쓰레딩과 동시성
CUDA 커널에서는 변수를 공유하고, 정점에 고유한 번호를 부여할 때 세마포어와 아토믹 연산을 이용해 동시성 문제를 해결한다.

## 데이터 취합과 출력
마지막으로, 각 CUDA 블록에서 처리한 데이터를 취합하여 최종적으로 OBJ 파일 형식으로 출력한다.
이렇게 구현된 알고리즘을 통해 3D 오브젝트 세그멘테이션이 완료된다.

# 프로그램 실행 결과
## TestCube.obj (Mode: parallel)
![R1](https://github.com/unta1337/Multi-Segmenter/assets/58779799/d93e99e1-a736-40fa-a9b6-f7dfa0a298c0) 

## Cube_noised.obj (Mode: parallel)
![R2](https://github.com/unta1337/Multi-Segmenter/assets/58779799/b5bb4fda-5980-413d-87a2-fde43be05c59) 

## Dodecahedron.obj (Mode: parallel)
![R3](https://github.com/unta1337/Multi-Segmenter/assets/58779799/89de4cdd-24ed-4a42-8a93-72727797a3c8)

## Icosahedron.obj (Mode: parallel)
![R4](https://github.com/unta1337/Multi-Segmenter/assets/58779799/d4a3b4a7-2f44-4a68-8343-4ab166a9a5a5) 

## Thingy.obj 앞면 (Mode: parallel)
![R5](https://github.com/unta1337/Multi-Segmenter/assets/58779799/42b74b41-7b78-4c79-8ccd-3acd6e0b4053) 

## Thingy.obj 뒷면 (Mode: parallel)
![R6](https://github.com/unta1337/Multi-Segmenter/assets/58779799/3853ae3c-8ef3-4516-93c7-fbf1b3f4e7e9) 

## Cube_noised.obj (Mode: cuda)
![RR1](https://github.com/unta1337/Multi-Segmenter/assets/58779799/f040c89d-90e4-4719-9850-08340d8d7a58)

## Dodecahedron.obj (Mode: cuda)
![RR2](https://github.com/unta1337/Multi-Segmenter/assets/58779799/3d4b36c5-a32d-4ffc-9d4a-abc2849c35ea)


# 성능 비교
![Result](https://github.com/unta1337/Multi-Segmenter/assets/58779799/a072fe26-03d1-4359-bc03-337ca5852c3f)
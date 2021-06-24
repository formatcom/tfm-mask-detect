## Sistema de bajo coste para detectar personas con mascarilla y su temperatura a través de redes neuronales

***
### Manual de instalación en Fedora
***

#### Requisitos del sistema
- Python 3
***

#### 1.- Clonar proyecto
~~~
$ git clone https://github.com/formatcom/tfm-mask-detect.git
$ cd tfm-mask-detect
~~~

#### 2.- Crear entorno virtual
~~~
$ python3 -m venv env
$ source env/bin/activate
~~~

#### 3.- Instalar dependencias
~~~
$ pip install -r requirements.txt
~~~

#### 4.- Conectar maixcube al pc y ejecutar lo siguiente
~~~
$ kflash -p /dev/ttyUSB0 -b 1500000 maixpy_k210_minimum_v0.6.2_mask.kfpkg
~~~

#### 5.- Introducir sd con el archivo app.py en la raiz

***


### Manual de compilación del firmware en Fedora
***
~~~
POR ESCRIBIR
~~~








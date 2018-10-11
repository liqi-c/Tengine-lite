#CROSS_COMPILE=aarch64-linux-gnu-
SYSROOT:=$(shell pwd)/sysroot/ubuntu_rootfs

ifneq ($(CROSS_COMPILE),)
   SYSROOT_FLAGS:=--sysroot=$(SYSROOT) 
   SYSROOT_LDFLAGS:=-L/usr/lib/aarch64-linux-gnu -L/lib/aarch64-linux-gnu
   PKG_CONFIG_PATH:=$(SYSROOT)/usr/lib/aarch64-linux-gnu/pkgconfig
   export PKG_CONFIG_PATH
endif

CC=$(CROSS_COMPILE)gcc -std=gnu99 $(SYSROOT_FLAGS)
CXX=$(CROSS_COMPILE)g++ -std=c++11 $(SYSROOT_FLAGS)
LD=$(CROSS_COMPILE)g++ $(SYSROOT_FLAGS) $(SYSROOT_LDFLAGS)

BUILT_IN_LD=$(CROSS_COMPILE)ld

GIT_COMMIT_ID=$(shell git rev-parse HEAD)

COMMON_CFLAGS+=-Wno-ignored-attributes -Werror
COMMON_CFLAGS+=-I$(shell pwd)/include -g -Wall

GIT_COMMIT_ID=$(shell git rev-parse HEAD)
COMMON_CFLAGS+=-DGIT_COMMIT_ID=\"$(GIT_COMMIT_ID)\"

COMMON_CFLAGS+=-DCONFIG_MT_SUPPORT -fPIC
COMMON_CFLAGS+=-DCL_USE_DEPRECATED_OPENCL_1_2_APIS

export CC CXX CFLAGS BUILT_IN_LD LD LDFLAGS CXXFLAGS COMMON_CFLAGS 
export GIT_COMMIT_ID

MAKEFILE_CONFIG=$(shell pwd)/makefile.config

export MAKEFILE_CONFIG

include $(MAKEFILE_CONFIG)

MAKEBUILD=$(shell pwd)/scripts/makefile.build

BUILD_DIR?=$(shell pwd)/build
INSTALL_DIR?=$(shell pwd)/install
TOP_DIR=$(shell pwd)

export INSTALL_DIR MAKEBUILD TOP_DIR

APP_SUB_DIRS=tests

LIB_SUB_DIRS=src
LIB_SO=$(BUILD_DIR)/libtengine-lite.so

LIB_OBJS =$(addprefix $(BUILD_DIR)/, $(foreach f,$(LIB_SUB_DIRS),$(f)/built-in.o))

ifeq ($(CONFIG_UNIT_TEST),)
	COMMON_CFLAGS+=-fvisibility=hidden
endif

ifeq ($(CONFIG_ARCH_ARM32),y)
    export CONFIG_ARCH_ARM32
	COMMON_CFLAGS+=-mfp16-format=ieee -mfpu=neon-fp16
endif

ifneq ($(CONFIG_OPT_CFLAGS),)
    export CONFIG_OPT_CFLAGS
endif

SUB_DIRS=$(LIB_SUB_DIRS) $(APP_SUB_DIRS)

default: $(LIB_SO) $(APP_SUB_DIRS) 

build : default

ifneq ($(MAKECMDGOALS),clean)
	     $(APP_SUB_DIRS): $(LIB_SO)
endif

clean: $(SUB_DIRS)

install: $(APP_SUB_DIRS)
	@mkdir -p $(INSTALL_DIR)/include $(INSTALL_DIR)/lib

Makefile : $(MAKEFILE_CONFIG)
	@touch Makefile
	@$(MAKE) clean

$(LIB_OBJS): $(LIB_SUB_DIRS);

$(LIB_SO): $(LIB_OBJS) 
	$(CC) -o $@ -shared -Wl,-Bsymbolic -Wl,-Bsymbolic-functions $(LIB_OBJS) $(LIB_LDFLAGS) -lpthread

$(LIB_SUB_DIRS):
	@$(MAKE) -C $@  -f $(MAKEBUILD) BUILD_DIR=$(BUILD_DIR)/$@ $(MAKECMDGOALS)

$(APP_SUB_DIRS):
	@$(MAKE) -C $@  BUILD_DIR=$(BUILD_DIR)/$@ $(MAKECMDGOALS)



distclean:
	find . -name $(BUILD_DIR) | xargs rm -rf
	find . -name $(INSTALL_DIR) | xargs rm -rf

.PHONY: clean install $(SUB_DIRS) build

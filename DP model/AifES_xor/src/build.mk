## Standard behavior must be included here
INCLUDE_DIRS += $(SOURCE_PATH)/$(USRSRC)  # add user sources to include path
CPPSRC += $(call target_files,$(USRSRC_SLASH),*.cpp)
CSRC += $(call target_files,$(USRSRC_SLASH),*.c)

APPSOURCES=$(call target_files,$(USRSRC_SLASH),*.cpp)
APPSOURCES+=$(call target_files,$(USRSRC_SLASH),*.c)

INCLUDE_DIRS += .
INCLUDE_DIRS += ./basic/base/aialgo
INCLUDE_DIRS += ./basic/base/ailayer
INCLUDE_DIRS += ./basic/base/ailoss
INCLUDE_DIRS += ./basic/base/aimath
INCLUDE_DIRS += ./basic/base/aiopti
INCLUDE_DIRS += ./core

#add compile flag
CFLAGS += -DAIDEBUG_ENABLE_PRINTING


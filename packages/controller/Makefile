CRAZYFLIE_BASE := ../../vendor/crazyflie-firmware

OOT_CONFIG := $(PWD)/config

# ifndef STEDGEAI_CORE_DIR
# $(error STEDGEAI_CORE_DIR is not set. Please set it to the path of the STEdge AI Core SDK)
# endif

EXTRA_CFLAGS += -I../../stedgeai/Inc -Wno-error=double-promotion

include $(CRAZYFLIE_BASE)/tools/make/oot.mk

.PHONY: clean-all
clean-all:
	$(MAKE) clean
	rm -f *.o .*.cmd
	find src -type f \( -name '*.o' -o -name '.*.cmd' \) -delete

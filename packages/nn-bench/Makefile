CRAZYFLIE_BASE := ../../vendor/crazyflie-firmware

OOT_CONFIG := $(PWD)/app-config

EXTRA_CFLAGS += -I../../stedgeai/Inc -Wno-error=double-promotion

include $(CRAZYFLIE_BASE)/tools/make/oot.mk

.PHONY: clean-all
clean-all:
	$(MAKE) clean
	rm -f *.o .*.cmd .*.d
	find src -type f \( -name '*.o' -o -name '.*.cmd' -o -name '.*.d' \) -delete

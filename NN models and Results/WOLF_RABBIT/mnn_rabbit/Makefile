TCRESTHOMEDIR?=$(HOME)/t-crest
TCRESTINCLUDEDIR?=$(TCRESTHOMEDIR)/patmos/c
TCRESTLINKDIR?=$(TCRESTHOMEDIR)/patmos/tmp
TCRESTNOCINIT?=$(TCRESTHOMEDIR)/patmos/c/cmp/nocinit.c
FBINCLUDEDIR?=./include

PROJECT?=mnn_rabbit
FUNCTION?=mnn_response

BUILDDIR?=$(TCRESTHOMEDIR)/patmos/tmp
LIBNOC=$(BUILDDIR)/libnoc.a
LIBMP=$(BUILDDIR)/libmp.a
LIBCORETHREAD=$(BUILDDIR)/libcorethread.a
LIBETH=$(BUILDDIR)/libeth.a
LIBELF=$(BUILDDIR)/libelf.a

#Delay to global memory per request in cycles
T_DELAY_CYCLES?=21 #21 #83
#Global memory transfer time per burst in cycles
G_DELAY_CYCLES?=0 #56 #21
#Transfer size (burst size) of the global memory in bytes

B_SIZE?=16

$(TCRESTHOMEDIR)/patmos/tmp/$(PROJECT).elf: *.c *.h
	patmos-clang \
	-target patmos-unknown-unknown-elf \
	-O2 \
	-I $(TCRESTINCLUDEDIR) \
	-I $(TCRESTINCLUDEDIR)/libelf/ \
	-I ./ \
	-I ./libfixmath \
	-I . \
	-mpatmos-disable-vliw \
	-mpatmos-stack-base=0x4000000 \
	-mpatmos-shadow-stack-base=0x3800000 \
	-Xgold --defsym -Xgold __heap_end=0x3000000 \
	-Xgold -T \
	-Xgold $(TCRESTHOMEDIR)/patmos/hardware/spm_ram.t \
	-o $(TCRESTHOMEDIR)/patmos/tmp/$(PROJECT).elf \
	libfixmath/fix16.c \
	libfixmath/fix16_exp.c \
	nn_rabbit.c \
	mnn.c \
	main.c \
	$(TCRESTNOCINIT) \
	-L $(TCRESTLINKDIR) \
	-lm \
	-lmp \
	-lnoc \
	-lcorethread \
	-leth \
	-lelf \
	-D PROGS_PER_CORE=$(PROGS_PER_CORE) \
	-mserialize=$(TCRESTHOMEDIR)/patmos/tmp/$(PROJECT).pml

build: $(TCRESTHOMEDIR)/patmos/tmp/$(PROJECT).elf

wcet-config:
	platin pml-config \
	--target patmos-unknown-unknown-elf \
	-o $(TCRESTHOMEDIR)/patmos/tmp/$(PROJECT)config.pml \
	--tdelay $(T_DELAY_CYCLES) \
	--gtime $(G_DELAY_CYCLES) \
	--bsize $(B_SIZE)

wcet: build wcet-config
	platin wcet \
	-i $(TCRESTHOMEDIR)/patmos/tmp/$(PROJECT).pml \
	--enable-wca \
	--disable-ait \
	--stats \
	--binary $(TCRESTHOMEDIR)/patmos/tmp/$(PROJECT).elf \
	-e $(FUNCTION) \
	-o $(PROJECT).wca \
	--report $(PROJECT)_wcet_report.txt \
	-i $(TCRESTHOMEDIR)/patmos/tmp/$(PROJECT)config.pml

APPSOURCES = ../src/*.c  
EXECUTABLE = darknet.elf

PLATFORM = zc706
SDSFLAGS = -sds-pf ${PLATFORM} 

CC = sdscc ${SDSFLAGS}

CFLAGS = -O3 -c
CFLAGS += -MMD -MP -MF"$(@:%.o=%.d)"
CFLAGS +=-Wall -Wfatal-errors
LFLAGS = -lm -O3

OBJDIR=./obj/
SRCDIR=../src/

#OBJECTS := $(APPSOURCES:.c=.o)
OBJ=gemm.o utils.o cuda.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o darknet.o detection_layer.o captcha.o route_layer.o writing.o box.o nightmare.o normalization_layer.o avgpool_layer.o coco.o dice.o yolo.o detector.o layer.o compare.o classifier.o local_layer.o swag.o shortcut_layer.o activation_layer.o rnn_layer.o gru_layer.o rnn.o rnn_vid.o crnn_layer.o demo.o tag.o cifar.o go.o batchnorm_layer.o art.o region_layer.o reorg_layer.o super.o voxel.o tree.o

#OBJ=gemm.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard ../src/*.h) Makefile

#DEPS := $(OBJECTS:.o=.d)

.PHONY: all clean ultraclean

all: obj ${EXECUTABLE}

${EXECUTABLE}: ${OBJS}
	${CC} ${LFLAGS} $^ -o $@ 

#-include ${DEPS}

$(OBJDIR)%.o: $(SRCDIR)%.c $(DEPS)
	${CC} ${CFLAGS} -c $< -o $@

obj:
	mkdir -p obj

clean:
	${RM} -rf ${EXECUTABLE} ${OBJECTS} *.d obj

ultraclean: clean
	${RM} ${EXECUTABLE}.bit 
	${RM} -rf _sds sd_card obj

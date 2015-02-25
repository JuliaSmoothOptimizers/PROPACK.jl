include ../Make.inc

DOBJ = slanbpro.F sreorth.F sgetu0.F ssafescal.F sblasext.F \
       slansvd.F printstat.F sgemm_ovwr.F  \
       slansvd_irl.F sbsvd.F sritzvec.F s$(MGS).F second.F

all:
	$(FC) -shared $(FFLAG) $(DOBJ) $(LDFLAGS) -o libspropack.$(SLIB)

debug:
	$(FC) -shared -g $(FFLAG) $(DOBJ) $(LDFLAGS) -o libspropack.$(SLIB)

clean:
	rm *.$(SLIB)
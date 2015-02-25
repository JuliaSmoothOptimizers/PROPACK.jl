include ../Make.inc

DOBJ = dlanbpro.F dreorth.F dgetu0.F dsafescal.F dblasext.F \
       dlansvd.F printstat.F dgemm_ovwr.F \
       dlansvd_irl.F dbsvd.F dritzvec.F d$(MGS).F second.F

all:
	$(FC) -shared $(FFLAG) $(LDFLAGS) $(DOBJ) -o libdpropack.$(SLIB)

debug:
	$(FC) -shared -g $(FFLAG) $(LDFLAGS) $(DOBJ) -o libspropack.$(SLIB)

clean:
	rm *.$(SLIB)
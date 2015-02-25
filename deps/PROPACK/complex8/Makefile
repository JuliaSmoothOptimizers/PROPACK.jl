include ../Make.inc

DOBJ = clanbpro.F creorth.F cgetu0.F csafescal.F sblasext.F cblasext.F \
       clansvd.F printstat.F cgemm_ovwr.F sgemm_ovwr.F \
       clansvd_irl.F sbsvd.F critzvec.F c$(MGS).F  second.F

all:
	$(FC) -shared $(FFLAG) $(DOBJ) $(LDFLAGS) -o libcpropack.$(SLIB)

debug:
	$(FC) -shared -g $(FFLAG) $(DOBJ) $(LDFLAGS) -o libspropack.$(SLIB)

clean:
	rm *.$(SLIB)
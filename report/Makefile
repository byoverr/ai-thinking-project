# Сборка отчёта: make -C report
ROOT := ..

.PHONY: pdf clean
pdf:
	cd $(ROOT) && tectonic -X compile report/REPORT.tex --outdir .

clean:
	rm -f $(ROOT)/REPORT.pdf

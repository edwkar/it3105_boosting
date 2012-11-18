TEXSOURCES=boosting-karlsen.tex

boosting-karlsen.pdf: $(TEXSOURCES) illustrations
	pdflatex $(@:pdf=tex)
#	bibtex $(@:pdf=aux)
#	pdflatex $(@:pdf=tex)
#	▷⋅while grep "Rerun to get" $(@:pdf=log); do pdflatex $(@:pdf=tex); done

illustrations: adaboost_acc_kr-vs-kp_dt3.png

adaboost_acc_kr-vs-kp_dt3.png: render_adaboost_acc_kr-vs-kp_dt3.m
	octave render_adaboost_acc_kr-vs-kp_dt3.m

clean: 
	rm -f *.aux *.log *.png *.pdf 

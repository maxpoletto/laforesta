TARGETS = laforesta.svg laforesta.html

laforesta.svg: genlogo.py
	python $< > $@

laforesta.html: genlogo.py
	python $< -html > $@

clean:
	@rm -f $(TARGETS) *~

TARGETS = laforesta.svg laforesta.html

laforesta.svg: genlogo.py
	python3 $< > $@

laforesta.html: genlogo.py
	python3 $< -html > $@

clean:
	@rm -f $(TARGETS) *~

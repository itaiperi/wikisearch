const dumpster = require('dumpster-dive');

options = {
	file: process.argv[2],
	db: 'simplewiki',
	skip_redirects: false,
	skip_disambig: false,
	batch_size: 1000,
	custom: function(doc) {
	    function capitalizeEnglish(word) {
	        letter = word[0]
            if((letter >= "a" && letter <= "z") || (letter >= "A" && letter <= "Z")) {
                return letter.toUpperCase() + word.substring(1);
            } else {
                return word;
            }
        }

        if(doc.isRedirect()) {
            if(doc.redirectTo() == undefined) {
                throw "Category redirect - Ignoring";
            } else {
                return {
                    title: doc.title(),
                    redirectTo: doc.redirectTo()['page'],
                }
            }
        }

		let links = doc.links().filter(link => {
			return !(link.type == 'external');
		}).map(link => capitalizeEnglish(link.page));
		links = new Set(links);
		links = [...links];
		return {
		    title: doc.title(),
		    text: doc.text(),
		    categories: doc.categories(),
		    links: links,
		};
	}
};
dumpster(options, () => console.log('Parsing is Done!'));

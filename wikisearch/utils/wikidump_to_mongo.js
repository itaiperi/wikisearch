const dumpster = require('dumpster-dive');

options = {
	file: process.argv[2],
	db: 'simplewiki',
	skip_redirects: false,
	skip_disambig: false,
	batch_size: 1000,
	workers: 1,
	custom: function(doc) {
	    function capitalizeEnglish(word) {
	        letter = word[0]
            if((letter >= "a" && letter <= "z") || (letter >= "A" && letter <= "Z")) {
                return letter.toUpperCase() + word.substring(1);
            } else {
                return word;
            }
        }

        redirect_wikipedia_internal_regex = [
            "H:.*", // Help
            "T:.*", // Template
            "MOS:.*", // Manual of Style
            "WT:.*", // Wikipedia talks
            "US:.*" // Hell knows...
        ]
        redirect_wikipedia_internal_regex = RegExp("^(" + redirect_wikipedia_internal_regex.join('|') + ")")

        if(doc.isRedirect()) {
            if(doc.redirectTo() == undefined) {
                throw "Category redirect - Ignoring";
            } else if (redirect_wikipedia_internal_regex.test(doc.title())) {
                // Filter redirections which relate to wikipedia-internal related pages (like explanations regarding help, templates, etc.)
                throw "Wikipedia-internal redirection - Ignoring";
            } else {
                return {
                    title: doc.title(),
                    // replace _ with a space, because of faults in the dump (redirections shouldn't have _), & with &amp;
                    redirectTo: doc.redirectTo()['page'].replace(/_/g, ' ').replace(/& /g, '&amp; '),
                }
            }
        }

        let links = doc.links().filter(link => {
            return !(link.type == 'external');
        // replace _ with a space, because of faults in the dump (links shouldn't have _), & with &amp;
		}).map(link => capitalizeEnglish(link.page).replace(/_/g, ' ').replace(/& /g, '&amp; '));
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

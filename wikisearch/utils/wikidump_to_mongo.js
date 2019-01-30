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

        function processTitle(title) {
            // replace _ with a space, because of faults in the dump (links shouldn't have _)
            // replace & with &amp; (unless it's &amp)
            // replace % with %25 to help decode handle % sign
            // decodeURIWithEncode is used to turn things like %20 to space, etc.
            return decodeURIComponent(capitalizeEnglish(title).replace(/_/g, ' ').replace(/&(?!amp)/g, '&amp;').replace(/%( |$)/, '%25$1').trim())
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
                    redirectTo: processTitle(doc.redirectTo()['page']),
                }
            }
        }

        let links = doc.links().filter(link => {
            return !(link.type == 'external');
        }).map(link => processTitle(link.page));
		links = new Set(links);
		links = [...links];

        return {
		    title: processTitle(doc.title()),
		    text: doc.text(),
		    categories: doc.categories(),
		    links: links,
		};
	}
};

dumpster(options, () => console.log('Parsing is Done!'));

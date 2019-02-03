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
            // decodeURIWithEncode is used to turn things like %20 to space, etc.
            return decodeURIComponent(capitalizeEnglish(title)
                // replace _ with a space, because of faults in the dump (links shouldn't have _)
                .replace(/_/g, ' ')
                // replace & with &amp; (unless it's &amp or &quot)
//                .replace(/&(?!(amp|quot))/g, '&amp;')
                .replace(/&amp;/g, '&')
                .replace(/&quot;/g, '"')
                // replace % with %25 to help decode handle % sign
                .replace(/%(?![0-9A-F][0-9A-F])/g, '%25')
                // replace multiple spaces with one space
                .replace(/  +/g, ' ')
                // get rid of leading ':'. No entry's title starts with ':', but it does appear in redirect/links because of dump structure
                .replace(/^:/, '')
                .trim())
        }

        redirect_wikipedia_internal_regex = [
            "H:.*", // Help
            "T:.*", // Template
            "MOS:.*", // Manual of Style
            "WT:.*", // Wikipedia talks
            "US:.*" // Hell knows...
        ]
        redirect_wikipedia_internal_regex = RegExp("^(" + redirect_wikipedia_internal_regex.join('|') + ")")

        if (redirect_wikipedia_internal_regex.test(doc.title())) {
            // Filter redirections which relate to wikipedia-internal related pages (like explanations regarding help, templates, etc.)
            throw "Wikipedia-internal redirection - Ignoring";
        }

        if(doc.isRedirect()) {
            if(doc.redirectTo() == undefined) {
                throw "Category redirect - Ignoring";
            } else {
                return {
                    title: processTitle(doc.title()),
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
		    categories: doc.categories().map(category => processTitle(category)),
		    links: links,
		};
	}
};

dumpster(options, () => console.log('Parsing is Done!'));

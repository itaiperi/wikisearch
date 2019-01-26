const dumpster = require('dumpster-dive');

options = {
	file: process.argv[2],
//	file: '/home/itai/Downloads/simplewiki_united-states_test.xml',
	db: 'simplewiki2',
	skip_redirects: false,
	skip_disambig: true,
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
//        function getMethods(obj) {
//            var result = [];
//            for (var id in obj) {
//                try {
//                    if (typeof(obj[id]) == "function") {
//                        result.push(id + ": " + obj[id].toString());
//                    }
//                } catch (err) {
//                    result.push(id + ": inaccessible");
//                }
//            }
//            return result;
//        }

        if(doc.isRedirect()) {
            return {
                title: doc.title(),
                redirectTo: doc.redirectTo(),
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

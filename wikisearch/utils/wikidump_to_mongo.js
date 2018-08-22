const dumpster = require('dumpster-dive')
options = {
	file: '/home/itai/Downloads/simplewiki-20180120-pages-articles-multistream.xml',
	db: 'simplewiki',
	custom: function(doc) {
		let links = doc.links().filter(link => {
			return !(link.type == 'external')
		}).map(link => link.page)
		links = new Set(links)
		links = [...links]
		return { _id: doc.title(), title: doc.title(), categories: doc.categories(), text: doc.text(), links: links }
	}
}
dumpster(options, () => console.log('Parsing is Done!'))
import spacy
from spacy import displacy
import pt_core_news_sm

# Load portugue language.
spacyPT = pt_core_news_sm.load()

# Tokenização (itemização)
entrada = spacyPT("Mais vale um asno que me carregue que um cavalo que me derrube.")
print( *entrada.text.split(), sep='\n' ) # não separou o ponto final.
print()

tk1 = [ token.text for token in entrada ] # separou o ponto final.
print( *tk1, sep='\n' )
print()

tk2 = [ token.text for token in entrada if not token.is_punct ] # excluiu o ponto final.
print( *tk2, sep='\n' )
print()

# Etiquetagem morfossintática (PoS tagging)
tk3 = [ (token.text, token.pos_) for token in entrada ]
print( *tk3, sep='\n' )
print()

# Reconhecimento de Entidades Nomeadas.
texto = spacyPT("A CBF fez um pedido de análise ao Comitê de Apelações da FIFA a fim de diminuir a pena do atacante Neymar, suspenso da Copa América pela Conmebol.")
print( texto.ents )
NER = [ (entidade,entidade.label_ ) for entidade in texto.ents ]
print( *NER, sep='\n' )
print()

# QUIZZES.
cap_3_bras_cubas = spacyPT( "Mas, já que falei nos meus dois tios, deixem-me fazer aqui um curto esboço genealógico.        O fundador de minha família foi um certo Damião Cubas, que floresceu na primeira metade do século XVIII. Era tanoeiro de ofício, natural do Rio de Janeiro, onde teria morrido na penúria e na obscuridade, se somente exercesse a tanoaria. Mas não; fez-se lavrador, plantou, colheu, permutou o seu produto por boas e honradas patacas, até que morreu, deixando grosso cabedal a um filho, o licenciado Luís Cubas. Neste rapaz é que verdadeiramente começa a série de meus avós -- dos avós que a minha família sempre confessou -  porque o Damião Cubas era afinal de contas um tanoeiro, e talvez mau tanoeiro, ao passo que o Luís Cubas estudou em Coimbra, primou no Estado, e foi um dos amigos particulares do vice-rei conde da Cunha.        Como este apelido de Cubas lhe cheirasse excessivamente a tanoaria, alegava meu pai, bisneto do Damião, que o dito apelido fora dado a um cavaleiro, herói nas jornadas da Africa, em prêmio da façanha que praticou arrebatando trezentas cubas ao mouros. Meu pai era homem de imaginação; escapou à tanoaria nas asas de um calembour. Era um bom caráter, meu pai, varão digno e leal como poucos. Tinha, é verdade, uns fumos de pacholice; mas quem não é um pouco pachola nesse mundo? Releva notar que ele não recorreu à inventiva senão depois de experimentar a falsificação; primeiramente, entroncou-se na família daquele meu famoso homônimo, o capitão-mor Brás Cubas, que fundou a vila de São Vicente, onde morreu em 1592, e por esse motivo é que me deu o nome de Brás. Opôs-se-lhe, porém, a família do capitão-mor, e foi então que ele imaginou as trezentas cubas mouriscas.        Vivem ainda alguns membros de minha família, minha sobrinha Venância, por exemplo, o lírio-do-vale, que é a flor das damas do seu tempo; vive o pai, o Cotrim, um sujeito que... Mas não antecipemos os sucessos; acabemos de uma vez com o nosso emplasto." )
q1 = [ ent.text for ent in cap_3_bras_cubas.ents if ent.label_ == 'PER' ]
print( *q1, sep='\n' )
print()

q2 = [ token.text for token in cap_3_bras_cubas if token.pos_ == 'DET' ]
print( *q2, sep='\n' )
print()

frases = [ frase for frase in cap_3_bras_cubas.sents ] 
print( frases[ 2 ] )
nlp = spacy.load( "en_core_web_sm" )
doc = nlp( frases[ 2 ].text )
displacy.serve( doc, style="dep" )
const tf = require('@tensorflow/tfjs');

const tensor = tf.tensor2d(
    [
        [1.0, 2.0], 
        [3.0, 4.0]
    ]
    );

console.log(tensor.toString());

console.log("-----------------------------------------------------------------------------")

//--------------------------------------------------------------------------------------------------------------


// Start at zero tensors
console.log('start', tf.memory().numTensors)
let keeper, chaser, seeker, beater
// Now we'll create tensors inside a tidy
tf.tidy(() => {
 keeper = tf.tensor([1,2,3])
 chaser = tf.tensor([1,2,3])
 seeker = tf.tensor([1,2,3])
 beater = tf.tensor([1,2,3])
 // Now we're at four tensors in memory
 console.log('inside tidy', tf.memory().numTensors)
 // protect a tensor
 tf.keep(keeper)
 // returned tensors survive
 return chaser
})
// Down to two
console.log('after tidy', tf.memory().numTensors)
keeper.dispose()
chaser.dispose()
// Back to zero
console.log('end', tf.memory().numTensors)
console.log("-----------------------------------------------------------------------------")

//--------------------------------------------------------------------------------------------------------

const users = ['Gant', 'Todd', 'Jed', 'Justin']
const bands = [
 'Nirvana',
 'Nine Inch Nails',
 'Backstreet Boys',
 'N Sync',
 'Night Club',
 'Apashe',
 'STP'
]
const features = [
 'Grunge',
 'Rock',
 'Industrial',
 'Boy Band',
 'Dance',
 'Techno'
]
// User votes
const user_votes = tf.tensor([
 [10, 9, 1, 1, 8, 7, 8],
 [6, 8, 2, 2, 0, 10, 0],
 [0, 2, 10, 9, 3, 7, 0],
 [7, 4, 2, 3, 6, 5, 5]
])
// Music Styles
const band_feats = tf.tensor([
 [1, 1, 0, 0, 0, 0],
 [1, 0, 1, 0, 0, 0],
 [0, 0, 0, 1, 1, 0],
 [0, 0, 0, 1, 0, 0],
 [0, 0, 1, 0, 0, 1],
 [0, 0, 1, 0, 0, 1],
 [1, 1, 0, 0, 0, 0]
])

// User's favorite styles
const user_feats = tf.matMul(user_votes, band_feats)
// Print the answers
user_feats.print()

// Let's make them pretty
const top_user_features = tf.topk(user_feats, features.length)
// Back to JavaScript
const top_genres = top_user_features.indices.arraySync()
// print the results
users.map((u, i) => {
 const rankedCategories = top_genres[i].map(v => features[v])
 console.log(u, rankedCategories)
})
console.log("-----------------------------------------------------------------------------")

